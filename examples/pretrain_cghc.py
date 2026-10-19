#!/usr/bin/env python3
"""Train a pre-norm Llama backbone with continuous generator hyper-connections (CGHC)."""

from __future__ import annotations

from typing import List

import pretrain_llama_base as base
from hyper_connections_common import disable_static_routing_weight_decay
from olmo_core.nn.transformer import (
    CGHCConfig,
    HyperConnectionsConfig,
    HyperConnectionsKind,
    TransformerBlockType,
)


def build_config(args, overrides: List[str]) -> base.ExperimentConfig:
    config = base.build_config(args, [])
    # Stream count, embedding expansion, mean collapse and sqrt(n) branch-output scaling
    # match the Liger mHC recipe; only the connector differs.
    cghc_config = HyperConnectionsConfig(
        kind=HyperConnectionsKind.cghc,
        num_residual_streams=4,
        scale_output_init_by_sqrt_n=True,
        collapse="mean",
        cghc=CGHCConfig(
            generator_type="conservative_diag_diss",
            # Preserve the mean-stream direction v (shared by every sublayer), decayed by 0.99 per
            # sublayer, and let each token pull mixed content into it with an input-dependent
            # inflow gamma in (-1, 1), starting at 0 (two-sided): alpha P + gamma P T Q + Q T Q.
            # With a shared v this is bounded at any depth; see docs/cghc_stability_notes.md.
            # Per-layer v instead: projection=v (add feedback_budget=<G_max> to cap the growth).
            projection="mean",
            projection_variant="inflow",
            projection_inflow=0.0,
            inflow_max=1.0,
            feedback_budget=None,
            projection_decay=0.99,
            dt=0.015,
            dt_min=0.0001,
            dt_max=0.35,
            dt_max_diss=0.35,
            learn_dt=True,
            vec_dt=True,
            sat_c=10,
            init_transition_contraction=0.95,
            bias=False,
            elementwise_affine=False,
            use_triton=True,
            # Temporal write-back adds token mixing that mHC lacks; enable it only as an ablation:
            # model.block.hyper_connections.cghc.writeback_kernel_sizes=[4,8,12]
            writeback_kernel_sizes=[],
            writeback_orthogonalize=True,
            # Read 1/n from each stream at init. The rotating one-stream read of mHC's gap-8 init
            # is model.block.hyper_connections.cghc.read_init=rotating (only tried with a 30-step
            # warmup, where it diverged).
            read_init="uniform",
        ),
    )
    config.model.block.name = TransformerBlockType.cghc
    config.model.block.hyper_connections = cghc_config
    # Apply connector overrides after the method config exists, then derive optimizer groups.
    config = config.merge(overrides)
    cghc_config = config.model.block.hyper_connections
    if (
        config.model.block.name != TransformerBlockType.cghc
        or cghc_config is None
        or cghc_config.kind != HyperConnectionsKind.cghc
    ):
        raise ValueError("pretrain_cghc.py requires a CGHC block and connector config")
    disable_static_routing_weight_decay(config, cghc_config)
    return config


def main() -> None:
    parser = base.build_parser()
    args, overrides = parser.parse_known_args()
    config = build_config(args, overrides)
    base.prepare_training_environment()
    try:
        base.train(config)
    finally:
        base.teardown_training_environment()


if __name__ == "__main__":
    main()
