"""Exercise the CGHC entry: config, parameter count, initialization and optimizer groups."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("hyperconnections.cghc")

from olmo_core.nn.hyper_connections import CGHCHyperConnection
from olmo_core.nn.transformer import HyperConnectionsKind, TransformerBlockType

ROOT = Path(__file__).resolve().parents[6]
SHAPE_CONFIGS = sorted((ROOT / "configs").glob("llama_400m_*.json"))
TINY_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 50280,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "max_sequence_length": 2048,
}


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    monkeypatch.chdir(ROOT)
    import pretrain_cghc

    return pretrain_cghc


def make_args(entry, tmp_path, model_config):
    data = tmp_path / "tokens.npy"
    np.save(data, np.zeros(4096, dtype=np.uint32))
    return entry.base.build_parser().parse_args(
        [
            f"--model-config={model_config}",
            f"--train-data-glob={data}",
            f"--eval-data-glob={data}",
            f"--save-folder={tmp_path / 'ckpt'}",
            "--wandb-project=",
            "--learning-rate=2e-3",
            "--global-train-batch-size=512",
            "--device-train-microbatch-size=8",
            "--sequence-length=2048",
            "--max-steps=7600",
            "--warmup-steps=760",
        ]
    )


def build_tiny(entry, tmp_path, overrides=(), device="cpu"):
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(TINY_CONFIG))
    config = entry.build_config(make_args(entry, tmp_path, path), list(overrides))
    model = config.model.build(init_device="meta")
    model.init_weights(device=torch.device(device), max_seq_len=256)
    return config, model


def connectors(model):
    return [m for m in model.modules() if isinstance(m, CGHCHyperConnection)]


@pytest.mark.parametrize("model_path", SHAPE_CONFIGS, ids=lambda p: p.name)
def test_shapes_use_cghc_recipe_and_count_params(entry, tmp_path, model_path):
    config = entry.build_config(make_args(entry, tmp_path, model_path), [])
    routing = config.model.block.hyper_connections
    assert config.model.block.name == TransformerBlockType.cghc
    assert routing.kind == HyperConnectionsKind.cghc
    assert routing.num_residual_streams == 4
    assert routing.reduce_mode == "mean"
    assert routing.scale_output_init_by_sqrt_n
    model = config.model.build(init_device="meta")
    assert model._hyper_connection_enabled
    assert config.model.num_params == model.num_params


def test_writeback_ablation_override_is_initialized(entry, tmp_path):
    _, model = build_tiny(
        entry, tmp_path, ["model.block.hyper_connections.cghc.writeback_kernel_sizes=[4,8]"]
    )
    for connector in connectors(model):
        writeback = connector.cghc.writeback
        assert writeback.kernel_sizes == (4, 8)
        for weight, kernel_size in zip(writeback.weights, writeback.kernel_sizes):
            assert torch.isfinite(weight).all()
            assert weight.abs().max() <= 1 / kernel_size**0.5


def test_cpu_forward_falls_back_to_pytorch(entry, tmp_path):
    _, model = build_tiny(entry, tmp_path)
    logits = model(torch.randint(0, 50280, (2, 32)))
    assert torch.isfinite(logits).all()
    assert not any(c.cghc._use_triton for c in connectors(model))


def test_meta_init_matches_cghc_initializer(entry, tmp_path):
    _, model = build_tiny(entry, tmp_path)
    found = connectors(model)
    assert len(found) == 2 * TINY_CONFIG["num_hidden_layers"]
    for connector in found:
        cghc = connector.cghc
        for name, tensor in [*cghc.named_parameters(), *cghc.named_buffers()]:
            assert torch.isfinite(tensor).all(), name
        # Zero-initialized predictor slices survive the generic nn.Linear reset.
        weight = cghc.input_proj.fused.weight
        for key, (start, end) in cghc.input_proj._slices.items():
            if cghc.input_proj._init[key] == "zero":
                assert weight[start:end].abs().max() == 0, key
            else:
                assert weight[start:end].std() < 0.02, key
                assert weight[start:end].abs().max() < 0.1, key  # no +/-2 clamp outliers
        # Connectors follow the model's parameter dtype (bf16), like HC / mHC.
        assert cghc.conserv_A.dtype == next(model.blocks["0"].attention.parameters()).dtype
        assert torch.allclose(cghc.conserv_A.float(), torch.eye(4), atol=0.05)
        assert torch.allclose(
            cghc.diss_diag.float(), torch.full((4,), cghc._diss_diag_init), rtol=1e-2
        )
        assert torch.allclose(cghc.alpha_read_in.float(), torch.tensor([0.01]), rtol=1e-2)
        if cghc.projection == "v":
            assert torch.allclose(cghc.base_projection_dir.float(), torch.full((4,), 0.5))
        elif cghc.projection == "mean":
            # Unit-norm mean direction plus 0.01 noise, refilled after to_empty().
            direction = cghc.projection_dir.float()
            assert torch.allclose(direction.norm(), torch.tensor(1.0), atol=1e-2)
            assert torch.allclose(direction, torch.full((4,), 0.5), atol=0.05)
        assert cghc.writeback is None
        # With zero predictor outputs the transition contracts by the nominal 0.95.
        cghc._use_triton = False
        x = torch.zeros(1, 4 * TINY_CONFIG["hidden_size"], dtype=weight.dtype)
        transition = cghc.compute_transition(cghc.input_proj(x)).float()
        assert torch.allclose(transition.diagonal(dim1=-2, dim2=-1), torch.tensor(0.95), atol=0.02)


@pytest.mark.parametrize("read_init", ["rotating", "uniform"])
def test_read_init(entry, tmp_path, read_init):
    _, model = build_tiny(
        entry, tmp_path, [f"model.block.hyper_connections.cghc.read_init={read_init}"]
    )
    for block_idx, block in model.blocks.items():
        for offset, connector in enumerate(
            (block.attention_hyper_connection, block.feed_forward_hyper_connection)
        ):
            read = torch.sigmoid(connector.cghc.read_in.float()[:, 0])
            if read_init == "uniform":
                assert connector.cghc.read_stream is None
                assert torch.allclose(read, torch.full((4,), 0.25), atol=0.01)
            else:
                # Attention reads stream (2 * block) % n and the FFN (2 * block + 1) % n, as mHC.
                selected = (2 * int(block_idx) + offset) % 4
                assert connector.cghc.read_stream == selected
                assert read.argmax() == selected and read[selected] > 0.999
                assert read.sum() < 1.01


def test_optimizer_groups_follow_cghc_decay_rule(entry, tmp_path):
    config, model = build_tiny(entry, tmp_path)
    optim = config.train_module.optim.build(model, strict=True)
    no_decay = {id(p) for g in optim.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    for connector in connectors(model):
        names = set(connector.no_decay_parameter_names())
        assert {"cghc.read_in", "cghc.write_out", "cghc.conserv_A"} <= names
        for name, param in connector.named_parameters():
            if name.startswith("cghc.module."):
                continue
            assert (id(param) in no_decay) == (name in names), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_forward_backward(entry, tmp_path):
    _, model = build_tiny(entry, tmp_path, device="cuda")
    input_ids = torch.randint(0, 50280, (2, 128), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), input_ids.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    for connector in connectors(model):
        for name, param in connector.named_parameters():
            if param.requires_grad:
                assert param.grad is not None and torch.isfinite(param.grad).all(), name


FEEDBACK = [
    "model.block.hyper_connections.cghc.projection=v",
    "model.block.hyper_connections.cghc.projection_variant=inflow",
    "model.block.hyper_connections.cghc.read_init=uniform",
]


def test_feedback_threads_gain_and_caps_it(entry, tmp_path):
    _, model = build_tiny(entry, tmp_path, FEEDBACK + ["model.block.hyper_connections.cghc.feedback_budget=1.5"])
    assert all(c.feedback for c in connectors(model))
    with torch.no_grad():  # request large inflow everywhere so the cap binds
        for c in connectors(model):
            c.cghc.input_proj.biases["inflow"].fill_(5.0)
    ids = torch.randint(0, 50280, (2, 16))
    h = model.expand_residual_streams(model.embeddings(ids))
    for block in model.blocks.values():
        h = block(h)
        assert isinstance(h, tuple) and h[1].shape == (2, 16)
        assert h[1].max() <= 1.5 + 1e-4
    logits = model(ids)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), ids.flatten()).backward()


def test_large_feedback_budget_matches_plain_inflow(entry, tmp_path):
    ids = torch.randint(0, 50280, (2, 16))
    torch.manual_seed(0)
    _, plain = build_tiny(entry, tmp_path, FEEDBACK)
    torch.manual_seed(0)
    _, fb = build_tiny(entry, tmp_path, FEEDBACK + ["model.block.hyper_connections.cghc.feedback_budget=1000000.0"])
    with torch.no_grad():
        torch.testing.assert_close(fb(ids), plain(ids))
