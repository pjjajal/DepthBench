# CGHC stream-mixing: norm analysis and the switched-system view

These notes cover only the stream-mixing part of a CGHC sublayer: how much the
connector can amplify the residual streams, for each projection variant, in one
sublayer and across depth.

## Setup

For each token and each feature dimension, the $n$ residual streams form a vector
$x \in \mathbb{R}^n$ ($n = 4$ in DepthBench). Each sublayer applies

$$x \leftarrow M x + u,$$

where $u$ is the branch output written into the streams and $M$ is the stream map.

- $T = \exp(A)$ is the transition. $A$ is a skew-symmetric (rotation) part minus a
  positive semidefinite (damping) part, both scaled by the step size `dt`. Because
  $\operatorname{sym}(A) \preceq 0$, we have $\lVert T \rVert_2 \le 1$.
- $v$ is the unit preserved direction, $P = vv^\top$ and $Q = I - P$.
- $\alpha \in [0, 1]$ is the projection decay (`projection_decay`).

We work in the basis $(v, v^\perp)$ and write $a = v^\top x$ and $q = Qx$, so
$\lVert x \rVert^2 = a^2 + \lVert q \rVert^2$. Norms are spectral norms.

The backward pass multiplies by $M^\top$, which has the same singular values, so
every bound below also applies to gradients.

## 1. Per-sublayer bounds

### Two facts used throughout

Split any $x$ into its component along $v$ and the rest:

$$x = a\,v + q, \qquad a = v^\top x, \qquad q = Qx \perp v, \qquad \lVert x \rVert^2 = a^2 + \lVert q \rVert^2.$$

Apply $T$ to the off-$v$ part $q$ and split the result the same way:

$$Tq = s\,v + Cq, \qquad s = v^\top T q \;(\text{a scalar}), \qquad C = QTQ.$$

The two pieces are orthogonal, and $\lVert T \rVert \le 1$, so

$$s^2 + \lVert Cq \rVert^2 = \lVert Tq \rVert^2 \le \lVert q \rVert^2. \tag{F}$$

Below, $s$ is the inflow ($T$ moving off-$v$ content into $v$), and $Cq$ is the part
that stays off $v$. Write $r^2 = \lVert q \rVert^2 - s^2 \ge 0$, so that
$\lVert Cq \rVert^2 \le r^2$ and $\lVert x \rVert^2 = a^2 + s^2 + r^2$.

### No projection

$M = T$, so $\lVert M \rVert \le 1$.

### Two-sided

$M = \alpha P + QTQ$. Using $Px = a\,v$ and $QTQx = QTq = Cq$:

$$Mx = \alpha a\,v + Cq.$$

The two terms are orthogonal, so

$$\lVert Mx \rVert^2 = \alpha^2 a^2 + \lVert Cq \rVert^2 \le \alpha^2 a^2 + \lVert q \rVert^2 \le a^2 + \lVert q \rVert^2 = \lVert x \rVert^2,$$

using (F) and $\alpha \le 1$. Hence $\lVert M \rVert \le 1$. In blocks,
$M = \operatorname{diag}(\alpha,\, QTQ)$: $v$ is completely isolated, with nothing
flowing in or out.

### Inflow (and one-sided as the case $\gamma = 1$)

$M = \alpha P + \gamma\, PTQ + QTQ$. Using $PTQx = (v^\top T q)\,v = s\,v$:

$$Mx = (\alpha a + \gamma s)\,v + Cq.$$

Again the two terms are orthogonal, so with (F):

$$\lVert Mx \rVert^2 = (\alpha a + \gamma s)^2 + \lVert Cq \rVert^2 \le (\alpha a + \gamma s)^2 + r^2.$$

The first term is a rank-one quadratic form in $(a, s)$:

$$(\alpha a + \gamma s)^2 = \begin{bmatrix} a \\ s \end{bmatrix}^{\!\top} \begin{bmatrix} \alpha^2 & \alpha\gamma \\ \alpha\gamma & \gamma^2 \end{bmatrix} \begin{bmatrix} a \\ s \end{bmatrix} \le (\alpha^2 + \gamma^2)(a^2 + s^2),$$

because that $2 \times 2$ matrix is $(\alpha, \gamma)^\top(\alpha, \gamma)$, whose
eigenvalues are $\alpha^2 + \gamma^2$ and $0$ (equivalently, Cauchy–Schwarz). So

$$\lVert Mx \rVert^2 \le (\alpha^2 + \gamma^2)(a^2 + s^2) + r^2 \le \max(1,\, \alpha^2 + \gamma^2)\,(a^2 + s^2 + r^2) = \max(1,\, \alpha^2 + \gamma^2)\,\lVert x \rVert^2,$$

that is,

$$\boxed{\;\lVert M \rVert \le \max\!\left(1,\; \sqrt{\alpha^2 + \gamma^2}\right)\;}$$

- **One-sided** is $\gamma = 1$: $\lVert M \rVert \le \sqrt{1 + \alpha^2} \le \sqrt{2}$.
  The bound is reached when $T$ rotates $q$ entirely into $v$ ($s = \lVert q \rVert$)
  and $(a, s) \propto (\alpha, \gamma)$; we measured about $1.40$ at $\alpha = 0.99$.
  The map has inflow into $v$ but no outflow, so it's non-normal and a single step
  can amplify.
- **Two-sided** is $\gamma = 0$: the bound is $\max(1, \alpha) = 1$, matching above.
- The bound depends only on $\gamma^2$, so the sign of $\gamma$ doesn't matter.
- The map **never amplifies when $\alpha^2 + \gamma^2 \le 1$**, i.e.
  $|\gamma| \le \sqrt{1 - \alpha^2}$, which is about $0.14$ at $\alpha = 0.99$.

In blocks, using an orthonormal basis $(v, e_2, \dots, e_n)$:

$$M = \begin{bmatrix} \alpha & \gamma\, b \\ 0 & C \end{bmatrix}, \qquad b = v^\top T Q \;(\text{a row vector}), \qquad \lVert b \rVert \le 1,\; \lVert C \rVert \le 1,$$

where the last two bounds hold because $\begin{bmatrix} b \\ C \end{bmatrix}$ is $TQ$
restricted to $v^\perp$, which has norm at most 1.

In the code, $\gamma = \gamma_{\max} \tanh(b_\gamma + w_\gamma^\top \hat x)$ for each
token, with $w_\gamma$ initialized to zero and $b_\gamma = \operatorname{atanh}(\gamma_0 / \gamma_{\max})$.

## 2. Products over depth

Let $\Pi_K = M_K \cdots M_1$ for $K$ sublayers ($K = 2L$ for $L$ blocks).

Norms are submultiplicative, $\lVert AB \rVert \le \lVert A \rVert \lVert B \rVert$, so

$$\lVert \Pi_K \rVert \le \prod_{k=1}^{K} \lVert M_k \rVert. \tag{S}$$

If every $\lVert M_k \rVert \le 1$, then $\lVert \Pi_K \rVert \le 1$ at any depth and
for any choice of $v_k$. That covers no projection, two-sided, and inflow with
$\alpha^2 + \gamma^2 \le 1$.

### Shared $v$ (`projection="mean"`)

With the same $v$ at every sublayer, every $M_k$ is upper-triangular in the same
basis:

$$M_k = \begin{bmatrix} \alpha & \gamma_k b_k \\ 0 & C_k \end{bmatrix}, \qquad \lVert b_k \rVert \le 1,\; \lVert C_k \rVert \le 1,\; |\gamma_k| \le \gamma_{\max}.$$

**The product stays triangular.** Suppose
$\Pi_{K-1} = \begin{bmatrix} \alpha^{K-1} & \beta_{K-1} \\ 0 & D_{K-1} \end{bmatrix}$.
Multiplying by $M_K$:

$$\Pi_K = \begin{bmatrix} \alpha & \gamma_K b_K \\ 0 & C_K \end{bmatrix} \begin{bmatrix} \alpha^{K-1} & \beta_{K-1} \\ 0 & D_{K-1} \end{bmatrix} = \begin{bmatrix} \alpha^K & \alpha\,\beta_{K-1} + \gamma_K b_K D_{K-1} \\ 0 & C_K D_{K-1} \end{bmatrix}.$$

So $D_K = C_K \cdots C_1$ (with $\lVert D_K \rVert \le 1$) and, starting from
$\beta_0 = 0$,

$$\beta_K = \alpha\,\beta_{K-1} + \gamma_K b_K D_{K-1} = \sum_{k=1}^{K} \alpha^{K-k}\, \gamma_k\, b_k\, D_{k-1}.$$

**Bounding the coupling block:**

$$\lVert \beta_K \rVert \le \sum_{k=1}^{K} \alpha^{K-k}\, |\gamma_k|\, \lVert b_k \rVert\, \lVert D_{k-1} \rVert \le \gamma_{\max} \sum_{j=0}^{K-1} \alpha^{j} = \gamma_{\max}\, \frac{1 - \alpha^K}{1 - \alpha} \le \frac{\gamma_{\max}}{1 - \alpha}.$$

**Bounding the whole product.** Split a block-triangular matrix into its diagonal and
off-diagonal parts:

$$\left\lVert \begin{bmatrix} \mu & \beta \\ 0 & \nu \end{bmatrix} \right\rVert \le \left\lVert \begin{bmatrix} \mu & 0 \\ 0 & \nu \end{bmatrix} \right\rVert + \left\lVert \begin{bmatrix} 0 & \beta \\ 0 & 0 \end{bmatrix} \right\rVert = \max(|\mu|, \lVert \nu \rVert) + \lVert \beta \rVert.$$

With $\mu = \alpha^K \le 1$ and $\nu = D_K$:

$$\boxed{\;\lVert \Pi_K \rVert \le 1 + \gamma_{\max}\,\frac{1 - \alpha^K}{1 - \alpha} \le 1 + \frac{\gamma_{\max}}{1 - \alpha}\quad\text{for every } K\;}$$

- At $\alpha = 1$ the sum is $K$, so the bound grows only linearly: $1 + K\gamma_{\max}$.
- The $v$ coordinate acts as a leaky accumulator of inflow. Nothing flows back out of
  it (the lower-left block stays $0$), so there's no feedback loop.
- The bound is loose: it doesn't drop to 1 when $\gamma$ is small, though by (S) it
  could.
- The current `mean` projection adds about $\pm 0.01$ noise per connector at
  initialization, so $v$ is only approximately shared and this bound holds only
  approximately.

### Per-layer $v$ (`projection="v"`)

The easiest way to see the difference is to track just two numbers through the
layers: how much of the stream lies **on** $v$ ($a$) and how much lies **off** it
($q$). To keep it simple, take a single off-$v$ direction, so both are scalars.

**One sublayer, in its own basis.** From the inflow map,

$$a_{\text{new}} = \alpha\, a + g\, q, \qquad q_{\text{new}} = c\, q,$$

where $g$ is the inflow (off-$v$ content moved into $v$) and $c$ is what stays off
$v$, with $g^2 + c^2 \le 1$. Content moves **from off-$v$ into $v$**, never the other
way.

**Shared $v$: no way back.** If every sublayer uses the same $v$, these two equations
simply repeat. $q$ never receives anything from $a$, so it can only shrink, and $a$ is
a decaying sum of what flowed in:

$$a_K = \alpha^K a_0 + \sum_k \alpha^{K-k}\, g_k\, q_{k-1}.$$

That sum is at most $\gamma_{\max}/(1-\alpha)$ times the input, which is the bounded
result above.

**Per-layer $v$: the basis change sends content back.** If the next sublayer's
direction is rotated by an angle $\theta$, the stream has to be re-split along the new
directions before that sublayer acts:

$$a' = \cos\theta\; a + \sin\theta\; q, \qquad q' = -\sin\theta\; a + \cos\theta\; q.$$

The $-\sin\theta\; a$ term moves content that was **on** $v$ to **off** $v$. Combined
with the inflow, which moves off-$v$ content back on, that closes a loop:

$$\text{on } v \;\xrightarrow{\;\text{new } v \text{ is rotated}\;}\; \text{off } v \;\xrightarrow{\;\text{inflow } g\;}\; \text{on } v \;\to\; \cdots$$

A shared $v$ has no such loop, because $\theta = 0$.

**How bad can it get?** Take the one-sided extreme, where $T$ moves everything off
$v$ into $v$ ($g = 1$, $c = 0$):

$$a_{\text{new}} = \alpha\, a + q, \qquad q_{\text{new}} = 0.$$

- The output lands entirely on $v$.
- The input that grows the most has $(a, q)$ proportional to $(\alpha, 1)$. It grows
  by a factor $\sqrt{1 + \alpha^2} \approx 1.41$, since $\alpha \cdot \alpha + 1 \cdot 1 = 1 + \alpha^2$
  is spread over an input of length $\sqrt{1 + \alpha^2}$.
- Now suppose the next sublayer's $v$ is rotated by about $45^\circ$ in the right
  direction (precisely $\theta = -\arctan(1/\alpha)$ with the signs above). Then the previous output, which lies on the old $v$, splits
  into exactly the $(\alpha, 1)$ mix that grows the most. So the next sublayer
  multiplies it by $1.41$ again.
- Repeat, and every sublayer multiplies by $1.41$:

$$\text{growth after } K \text{ sublayers} = (1 + \alpha^2)^{K/2}.$$

| Sublayers $K$ | per-layer $v$ (rotating $\approx 45^\circ$ each time) | same maps, shared $v$ |
|---|---|---|
| 2 | 1.98 | 1.39 |
| 4 | 3.92 | 1.37 |
| 12 | 60 | 1.26 |
| 24 | 3,633 | 1.12 |

(Computed numerically with $\alpha = 0.99$.) With a shared $v$, the same maps make the
stream shrink, because content that flows into $v$ then just decays.

**General bound.** For per-layer $v$, the only bound is the product of the per-layer
gains (S):

$$\lVert \Pi_K \rVert \le \max\!\left(1,\; \alpha^2 + \gamma_{\max}^2\right)^{K/2},$$

and the example above shows this can actually happen. In the model, $v$ is computed
from the input at each layer, so nothing prevents consecutive directions from rotating
the harmful way; whether they do depends on what training learns.

This matches what we observed. Two-sided was stable at every shape. One-sided and
inflow with per-layer $v$ diverged at L18, and one-sided at L12 had one large spike
and recovered.

### A depth-aware cap

To guarantee $\lVert \Pi_K \rVert \le C$ with per-layer $v$, it's enough by (S) that
every sublayer has gain at most $C^{1/K}$:

$$\sqrt{\alpha^2 + \gamma^2} \le C^{1/K} \iff \alpha^2 + \gamma^2 \le C^{2/K} = C^{1/L} \iff |\gamma| \le \gamma_{\max} = \sqrt{C^{1/L} - \alpha^2},$$

using $K = 2L$ (this needs $C^{1/L} \ge \alpha^2$, which holds for any $C \ge 1$).

| $\gamma_{\max}$ at $\alpha = 0.99$ | L12 | L18 | L24 | L70 |
|---|---|---|---|---|
| $C = 2$ | 0.28 | 0.24 | 0.22 | 0.17 |
| $C = 4$ | 0.38 | 0.32 | 0.28 | 0.20 |
| $C = 10$ | 0.48 | 0.40 | 0.35 | 0.23 |

Check: at L12 with $C = 4$, each sublayer's gain is
$\sqrt{0.99^2 + 0.377^2} \approx 1.059$, and $1.059^{24} \approx 4.0$.

### Writes

With $\lVert \Pi \rVert \le C$, the full update gives
$\lVert x_K \rVert \le C \lVert x_0 \rVert + C \sum_k \lVert u_k \rVert$. The branch
writes add up linearly with depth, as in any residual network.

## 3. The switched-system view

The stack $x_{k+1} = M_k x_k$, with $M_k$ chosen from a set $\mathcal{M}$ (depending
on the input and the layer), is a switched linear system. The stability question is
whether all products stay bounded.

**Joint spectral radius.**
$\rho(\mathcal{M}) = \limsup_k \sup \lVert M_k \cdots M_1 \rVert^{1/k}$. Bounded
products imply $\rho \le 1$.

**Extremal norm.** For a bounded family, all products are bounded if and only if
there is a norm $\lVert \cdot \rVert_*$ in which every $M \in \mathcal{M}$ satisfies
$\lVert M \rVert_* \le 1$ (Rota–Strang; Barabanov).

**Quadratic sufficient condition (a common Lyapunov function).** If
$M^\top P M \preceq P$ for every $M \in \mathcal{M}$, with $P \succ 0$, then
$\lVert \Pi \rVert_2 \le \sqrt{\kappa(P)}$. Equivalently, in coordinates
$y = P^{1/2} x$, every map is non-amplifying.

**Changing norm along the depth.** If $M_k^\top P_k M_k \preceq P_{k-1}$ with
$cI \preceq P_k \preceq CI$, then $\lVert \Pi \rVert \le \sqrt{C/c}$. In factored
form, $M_k = L_k^{-1} \tilde T_k L_{k-1}$ with $\lVert \tilde T_k \rVert \le 1$ gives

$$\Pi_K = L_K^{-1} \left(\tilde T_K \cdots \tilde T_1\right) L_0,$$

so the intermediate $L$'s cancel. Each layer's choice depends on the previous layer's
metric. The inverse can be avoided by choosing $L$ from a family with a cheap inverse:
diagonal, $\exp(K)$, or orthogonal times diagonal.

**Switching between stable systems can be unstable.** This is the per-layer-$v$
failure: each one-sided map is benign in its own basis, but the family has no common
norm. Dwell-time results say stability holds if switching is slow enough, which
suggests regularizing how quickly $v$ changes across depth.

**Building the guarantee into the generator.** With
$A = W^{-1}(J - R)$, $J$ skew and $R \succeq 0$, we get
$A^\top W + WA = -2R \preceq 0$, so $\exp(A)$ is non-amplifying in the $W$-norm. This
is the port-Hamiltonian (contraction-metric) form; CGHC's
conservative-minus-dissipative generator is the case $W = I$.

### What follows

- **A shared metric adds no capacity.** With the same $W$ in every layer, the stack is
  ordinary CGHC in the coordinates $y = Lx$. $L$ only touches the boundaries: the
  expand and collapse vectors and the reads and writes, which are learned or fixed
  anyway.
- **Within bounded products, amplification is an artifact of coordinates.** By the
  extremal-norm theorem, every bounded-product design is non-amplifying in some
  coordinates. Allowing amplification therefore doesn't add expressiveness; what
  differs between designs is the parameterization, meaning which reads, writes, gate
  ranges and boundary vectors are reachable or easy to reach.
- **Working hypothesis.** One-sided's early advantage likely came from working around
  the bounded sigmoid gates (reads in $(0, 1)$, writes in $(0, 2)$), not from
  amplification. The runs with wider gate ranges ($(0, 4)$) and signed tanh gates
  ($(-4, 4)$) test this.
- **Reads vs writes.** The read feeds into a pre-norm, so its overall scale cancels;
  only the relative weights across streams, and with tanh their signs, survive.
  Writes go into the streams without normalization, so the write range is more likely
  to matter.

## 4. Feedback budget controllers (`cghc_fb`)

Per-layer $v$ lets each sublayer amplify by up to $\sqrt{\alpha^2 + \gamma^2}$, and those
gains can compound (Section 2). The feedback variants keep per-layer $v$ but cap the total
growth directly: each token carries one number through the depth, and every connector limits
its inflow so that number never exceeds a budget.

### The loop

Depth plays the role of time; step $k$ is sublayer $k$.

| Control concept | In CGHC feedback |
|---|---|
| Process | the stream through sublayer $k$: $x_{k+1} = M_k x_k + u_k$ |
| Actuator | the inflow weight $\gamma_k$ |
| Command | $\gamma_k^{\text{req}}$, the inflow the model requests |
| Disturbances | everything else that is learned: $T_k$, $v_k$, the writes $u_k$ |
| State | $G_k$, a running upper bound on $\lVert M_k \cdots M_1 \rVert$ |
| Constraint | $G_k \le G_{\max}$ (`feedback_budget`) |
| Controller | clip $\gamma_k$ so the next step cannot break the constraint |

Each connector receives $G_{k-1}$ (the first starts from $G_0 = 1$), clips its inflow, mixes,
and passes on $G_k$. In DepthBench, blocks pass `(streams, G)` to each other, with $G$ of shape
`[batch, seq]`. $G$ is detached: it is controller state, not something the model trains through.

### Controller 1: fixed charge

By Section 1, an inflow sublayer satisfies
$\lVert M_k \rVert \le \max\!\left(1, \sqrt{\alpha^2 + \gamma_k^2}\right)$.
The controller charges each step this bound.

1. Room left: $r_k = G_{\max} / G_{k-1} \ge 1$.
2. Clip: $|\gamma_k| \le \sqrt{r_k^2 - \alpha^2}$, which is exactly the condition
   $\max\!\left(1, \sqrt{\alpha^2 + \gamma_k^2}\right) \le r_k$.
3. Charge: $G_k = G_{k-1} \cdot \max\!\left(1, \sqrt{\alpha^2 + \gamma_k^2}\right)$.

**Guarantee.** By submultiplicativity,

$$\lVert M_k \cdots M_1 \rVert \le \prod_{j \le k} \lVert M_j \rVert \le G_k \le G_{\max}$$

for every $k$, at any depth and for any sequence of $v_k$. The backward pass multiplies by the
transposes, which have the same norms, so the same cap holds there.

**Behavior.**
- At initialization $\gamma = 0$, so every step charges 1 and the controller is inactive.
- Once the budget is spent ($r_k = 1$), the cap is $\sqrt{1 - \alpha^2} \approx 0.14$ at
  $\alpha = 0.99$: inflow is not turned off, but held at the largest value that never
  amplifies.
- Each charge is at least 1, so the budget never comes back. Early sublayers can use it up and
  leave none for later ones.

Example with $G_{\max} = 2$, $\alpha = 0.99$ and full requested inflow ($\gamma^{\text{req}} = 1$):

| Sublayer | $G_{k-1}$ | room $r_k$ | cap | $\gamma_k$ | charge | $G_k$ |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 2.00 | 1.73 | 1.00 | 1.41 | 1.41 |
| 2 | 1.41 | 1.42 | 1.01 | 1.00 | 1.41 | 2.00 |
| 3 and later | 2.00 | 1.00 | 0.14 | 0.14 | 1.00 | 2.00 |

### Controller 2: refill from damping (diagonal dissipation)

Controller 1 charges at least 1 even when the step damps the stream. Controller 2 charges a
damping-aware bound instead, so damped steps give budget back.

**A free bound on $\lVert T \rVert$.** With `conservative_diag_diss`, the generator is

$$A = S - \operatorname{diag}(\delta_1, \dots, \delta_n), \qquad S^\top = -S, \qquad \delta_i = \mathrm{dt}_{\text{diss},i}\, d_i \ge 0.$$

(The `vec_dt` sandwich keeps $S$ skew-symmetric.) The symmetric part of $A$ is therefore exactly
$-\operatorname{diag}(\delta_i)$. For $y(t) = e^{tA} x$,

$$\frac{d}{dt}\lVert y \rVert^2 = 2\, y^\top A\, y = -2 \sum_i \delta_i\, y_i^2 \le -2\, \delta_{\min} \lVert y \rVert^2,$$

so $\lVert y(1) \rVert \le e^{-\delta_{\min}} \lVert x \rVert$, that is,

$$\lVert T \rVert \le \tau = \exp\!\left(-\min_i \delta_i\right).$$

This is a minimum over the $n$ damping rates CGHC already computes, so it costs essentially
nothing. It is exact as a bound, but $\lVert T \rVert$ itself is usually smaller, because the
rotation keeps moving content out of the least-damped stream.

**The inflow bound with $\tau$.** Redo the Section 1 argument with $\lVert T \rVert \le \tau$.
Now $s^2 + \lVert Cq \rVert^2 \le \tau^2 \lVert q \rVert^2$, so write
$r^2 = \tau^2 \lVert q \rVert^2 - s^2$. Then

$$\lVert Mx \rVert^2 \le (\alpha a + \gamma s)^2 + r^2, \qquad \lVert x \rVert^2 = a^2 + \frac{s^2 + r^2}{\tau^2}.$$

Substituting $s = \tau s'$ and $r = \tau r'$ turns the constraint into $a^2 + s'^2 + r'^2 = 1$
and the objective into $(\alpha a + \gamma \tau s')^2 + \tau^2 r'^2$. The same rank-one argument
as before gives

$$\lVert M \rVert \le \max\!\left(\tau,\; \sqrt{\alpha^2 + \gamma^2 \tau^2}\right).$$

With $\tau = 1$ this is controller 1's bound.

**The controller.**

1. Room left: $r_k = G_{\max} / G_{k-1}$.
2. Clip: $|\gamma_k| \le \sqrt{r_k^2 - \alpha^2}\,/\,\tau_k$.
3. Charge: $G_k = G_{k-1} \cdot \max\!\left(\tau_k,\; \sqrt{\alpha^2 + \gamma_k^2 \tau_k^2}\right)$.

The guarantee is the same as for controller 1, because each charge is still an upper bound on
$\lVert M_k \rVert$.

**How much it refills.** $M_k v = \alpha v$, so $\lVert M_k \rVert \ge \alpha$ and no charge is
below $\alpha$. At $\alpha = 0.99$ each step returns at most about 1% of budget, so recovering
from a large early spend takes tens of sublayers. A step gives budget back only when every
stream is damped ($\tau_k < 1$) and its inflow is small enough that
$\sqrt{\alpha^2 + \gamma_k^2 \tau_k^2} < 1$.

**Why only diagonal dissipation.** The bound needs the smallest eigenvalue of the damping
matrix. For `diag_diss` that is a minimum over $n$ numbers. For `psd_diss` or `laplacian` it is
a per-token eigenvalue problem.

### Caveats

- **Exact arithmetic.** $T$ and the stream map are rounded to the training dtype (bf16) before
  mixing, which can put a step's true norm slightly above its charge. Accounting for this
  rigorously would cost more than the refill returns, so neither controller does.
- **The guarantee covers the stream maps.** $\gamma_k$ depends on the input, so the full
  Jacobian has small extra terms the norm argument does not include, and the branch writes
  still add up across depth as in any residual network.
- **Conservative.** $G$ is a worst-case bound, so the controller can clip inflow even when the
  actual stream is not growing that much.

### Settings

`CGHCConfig` fields (all require `projection_variant="inflow"`):

| Field | Meaning |
|---|---|
| `feedback_budget` | $G_{\max}$; `None` turns the feedback off |
| `feedback_refill` | `True` for controller 2 (diagonal dissipation only) |
| `inflow_max`, `projection_inflow` | range and initial value of the requested inflow, as for plain inflow |

## Reading

- R. Jungers, *The Joint Spectral Radius: Theory and Applications* (Springer, 2009),
  chapter 1.
- D. Liberzon, *Switching in Systems and Control* (Birkhäuser, 2003).
- S. Boyd, L. El Ghaoui, E. Feron, V. Balakrishnan, *Linear Matrix Inequalities in
  System and Control Theory* (SIAM, 1994).
- L. N. Trefethen and M. Embree, *Spectra and Pseudospectra* (Princeton, 2005).
- W. Lohmiller and J.-J. Slotine, "On contraction analysis for non-linear systems",
  *Automatica* (1998).
- M. Revay, R. Wang, I. Manchester, "Recurrent Equilibrium Networks: Flexible Dynamic
  Models with Guaranteed Stability and Robustness", *IEEE Transactions on Automatic
  Control* (2023).

Citations are from memory; check details before citing.
