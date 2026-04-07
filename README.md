# PDE_D

# Deriving Diffusion Coefficient D via Numerical PDE Inversion

## Full Mathematical Framework for Comparing Oxygen Delivery Materials

---

## 1. The Physical Problem

You have an oxygen-carrier material (e.g. PFCMC) loaded with O₂ at its core.
Over time, oxygen diffuses outward through the material cross-section.
You measure the oxygen concentration profile C(x, t) along a 1D diagonal section
at discrete positions x ∈ {x₁, ..., x_N} and times t ∈ {t₀, t₁, ..., t_M},
where Δt = 5 min.

**Goal**: Extract D (mm²/s) and a retention metric to rank materials.

---

## 2. Governing Equation

Fick's Second Law of Diffusion in 1D:

$$
\frac{\partial C}{\partial t} = D \, \frac{\partial^2 C}{\partial x^2} \tag{1}
$$

This is a linear parabolic PDE. It says the *rate of change* of concentration
at any point equals D times the *curvature* of the concentration profile at
that point.

- Where the profile is concave down (∂²C/∂x² < 0, i.e. at the peak), C decreases.
- Where the profile is concave up (∂²C/∂x² > 0, i.e. at the edges), C increases.
- The rate of all this is governed by D.

---

## 3. Discretisation: Crank-Nicolson Method

### 3.1 Why Crank-Nicolson?

Three options exist for discretising Eq.(1) in time:

| Method | Scheme | Stability | Accuracy |
|---|---|---|---|
| Explicit (Forward Euler) | C^(n+1) from C^n directly | Requires Δt < Δx²/(2D) — **unstable for your data** | O(Δt, Δx²) |
| Implicit (Backward Euler) | Solve linear system | Unconditionally stable | O(Δt, Δx²) |
| **Crank-Nicolson** | **Average of explicit + implicit** | **Unconditionally stable** | **O(Δt², Δx²)** |

Your data has Δx = 0.0345 mm and Δt = 300 s. For D ≈ 10⁻³ mm²/s, the explicit
stability limit would be Δt < (0.0345)²/(2 × 10⁻³) ≈ 0.6 s — you'd need 500
sub-steps per measurement interval. Crank-Nicolson has no such restriction AND
is second-order accurate in time.

### 3.2 The Scheme

Evaluate the spatial derivative as the average of timesteps n and n+1:

$$
\frac{C_i^{n+1} - C_i^n}{\Delta t}
= \frac{D}{2} \left[
\frac{C_{i-1}^{n+1} - 2C_i^{n+1} + C_{i+1}^{n+1}}{\Delta x^2}
+ \frac{C_{i-1}^{n} - 2C_i^{n} + C_{i+1}^{n}}{\Delta x^2}
\right] \tag{2}
$$

Define the mesh ratio:

$$
r = \frac{D \, \Delta t}{2 \, \Delta x^2} \tag{3}
$$

Rearranging (2), group unknowns (n+1) on the left, knowns (n) on the right:

$$
-r \, C_{i-1}^{n+1} + (1 + 2r) \, C_i^{n+1} - r \, C_{i+1}^{n+1}
= r \, C_{i-1}^{n} + (1 - 2r) \, C_i^{n} + r \, C_{i+1}^{n} \tag{4}
$$

### 3.3 Matrix Form

For interior points i = 1, 2, ..., N-2 (with i=0 and i=N-1 as boundaries),
this is a tridiagonal linear system:

$$
\mathbf{A} \, \mathbf{C}^{n+1} = \mathbf{B} \, \mathbf{C}^n + \mathbf{b}^n \tag{5}
$$

where:

```
        ⎡ 1+2r   -r              ⎤         ⎡ 1-2r    r              ⎤
        ⎢  -r   1+2r  -r         ⎥         ⎢   r   1-2r   r         ⎥
  A  =  ⎢        -r  1+2r  -r    ⎥,   B =  ⎢         r  1-2r   r    ⎥
        ⎢             ...        ⎥         ⎢             ...        ⎥
        ⎣              -r  1+2r  ⎦         ⎣               r  1-2r  ⎦
```

The vector **b**ⁿ accounts for boundary conditions (known values at edges).

### 3.4 Boundary Conditions

**Dirichlet (fixed-value) BCs** — we pin the edges to measured values:

$$
C(x_0, t) = C_{\text{left}}(t), \quad C(x_{N-1}, t) = C_{\text{right}}(t) \tag{6}
$$

In practice, we use the average of the first/last 10 pixels (smoothed edges)
to reduce noise. This is the correct physical BC because your material
exchanges O₂ with the surrounding environment at its boundaries.

### 3.5 Thomas Algorithm (Tridiagonal Solve)

The tridiagonal system (5) is solved in O(N) operations via forward elimination
and back-substitution:

**Forward sweep** (i = 1, ..., N-2):

$$
c'_i = \frac{c_i}{b_i - a_i \, c'_{i-1}}, \quad
d'_i = \frac{d_i - a_i \, d'_{i-1}}{b_i - a_i \, c'_{i-1}} \tag{7}
$$

**Back-substitution** (i = N-3, ..., 0):

$$
C_i^{n+1} = d'_i - c'_i \, C_{i+1}^{n+1} \tag{8}
$$

---

## 4. The Inverse Problem: Finding D

### 4.1 Forward Problem

Given D, we can compute C_sim(x, t; D) by marching Eq.(4) forward from
the initial condition C(x, t₀) = measured first profile.

### 4.2 Objective Function

Define the misfit between simulation and data:

$$
J(D) = \sum_{n=1}^{M} \sum_{i=m}^{N-m}
\left[ C_{\text{sim}}(x_i, t_n; D) - C_{\text{data}}(x_i, t_n) \right]^2 \tag{9}
$$

where m is a margin (we exclude edge pixels to avoid BC bias).

### 4.3 Optimisation

D is a single scalar. The objective J(D) is smooth and unimodal
(verified by scanning D over orders of magnitude). We minimise with
bounded scalar optimisation:

$$
D^* = \underset{D \in [10^{-6}, \, 5 \times 10^{-2}]}{\arg\min} \; J(D) \tag{10}
$$

Using `scipy.optimize.minimize_scalar` with Brent's method (bounded).

### 4.4 Quality Metric

$$
\text{RMSE} = \sqrt{\frac{J(D^*)}{(N - 2m)(M - 1)}} \tag{11}
$$

This is in the same units as C (% Air Saturation), directly interpretable
as "how well does simple Fickian diffusion explain your data."

---

## 5. Why the PDE Method is Superior for Material Comparison

### 5.1 The Moment Method's Limitations for Your Data

The moment method (D = slope(σ²)/2) assumes:

1. **Unbounded domain** — your material is finite (12.5 mm). Once the
   diffusion front hits the edges, σ² saturates and the method fails.
   In your data, σ² plateaus after ~100 min.

2. **Conserved total mass** — your data shows M₀ dropping from 414 to
   ~180 (O₂ leaves through boundaries). The variance of a shrinking
   distribution is not governed by dσ²/dt = 2D alone.

3. **Positive excess everywhere** — at late times, edge concentrations
   exceed the initial baseline, making "excess" negative at boundaries.
   This corrupts the moment integrals.

4. **Only uses early data** — restricted to the first ~60 min out of
   485 min of data. Throws away 88% of your measurements.

### 5.2 The PDE Method's Advantages

| Property | Moment Method | PDE Method |
|---|---|---|
| Uses all timesteps | No (early only) | **Yes (all 98)** |
| Handles finite domains | No | **Yes (Dirichlet BCs)** |
| Handles O₂ loss at edges | No | **Yes (measured BCs)** |
| Handles late-time flattening | No | **Yes** |
| Assumes profile shape | No (good) | No (good) |
| Model-free | Yes | No (assumes Fick's law) |
| Output | Single number | **D + full predicted C(x,t)** |
| Residual diagnostics | R² on σ² only | **Per-pixel, per-timestep RMSE** |

### 5.3 Why This Matters for Comparing Materials

When you compare Material A vs Material B, you need a D that is:

1. **Consistent across conditions**: The PDE method uses ALL your data,
   so it averages over noise and gives a single robust D even if individual
   timesteps are noisy.

2. **Physically correct at boundaries**: Your materials exchange O₂ with
   the environment. Ignoring this (as the moment method does) systematically
   biases D.

3. **Accompanied by a goodness-of-fit**: The RMSE tells you whether
   Fick's law even applies. If Material A has RMSE = 5 and Material B
   has RMSE = 25, then Material B may have concentration-dependent D
   or reaction kinetics — important to know!

4. **Reproducible**: The PDE method has no subjective choices about
   "how many early timesteps to include." Same data → same D, always.

---

## 6. The Retention Metric: τ_half

For ranking "which material stays oxygenated longest," define:

$$
C_{\text{peak}}^{\text{ex}}(t) = \max_x \left[ C(x,t) - C_\infty \right] \tag{12}
$$

$$
\tau_{1/2} = \text{time at which} \quad
C_{\text{peak}}^{\text{ex}}(\tau_{1/2}) = \frac{1}{2} \, C_{\text{peak}}^{\text{ex}}(0) \tag{13}
$$

**τ₁/₂ is the half-life of the peak oxygen excess.**

### 6.1 Why τ₁/₂ and Not Just 1/D?

For identical geometry, τ₁/₂ ∝ 1/D, so lower D means longer retention.
But τ₁/₂ also depends on:

- **Initial loading width** (σ₀): A broader initial loading decays
  slower even with the same D.
- **Material thickness**: Thicker samples retain longer.
- **Boundary conditions**: Materials with less permeable surfaces
  retain longer.

τ₁/₂ captures ALL of these effects in a single number. D alone does not.

### 6.2 For an Ideal Gaussian in Unbounded Medium

If C(x,0) is Gaussian with variance σ₀², the peak decays as:

$$
C_{\text{peak}}(t) \propto \frac{1}{\sqrt{\sigma_0^2 + 2Dt}} \tag{14}
$$

Setting this to half its initial value:

$$
\frac{1}{\sqrt{\sigma_0^2 + 2D\tau}} = \frac{1}{2} \cdot \frac{1}{\sqrt{\sigma_0^2}}
$$

$$
\tau_{1/2} = \frac{3 \, \sigma_0^2}{2 \, D} \tag{15}
$$

But your data is NOT an ideal Gaussian in an unbounded medium. That's why
we measure τ₁/₂ directly from the data rather than computing it from Eq.(15).

---

## 7. Complete Workflow for Material Comparison

For each material sample:

```
1. Acquire C(x, t) cross-section data at 5-min intervals
2. Detect loading→diffusion transition (auto: peak of centre-region mean)
3. Trim loading phase
4. Run PDE inversion → D*, RMSE
5. Measure τ₁/₂ from peak excess decay
6. Record: D*, RMSE, τ₁/₂, initial peak, FWHM₀
```

### Comparison table (example):

| Material | D (mm²/s) | RMSE | τ₁/₂ (min) | FWHM₀ (mm) | Verdict |
|---|---|---|---|---|---|
| PFCMC-A | 5.9 × 10⁻⁴ | 3.2 | 85 | 4.3 | Best retention |
| PFCMC-B | 1.2 × 10⁻³ | 4.1 | 52 | 4.1 | Moderate |
| Hydrogel-C | 3.5 × 10⁻³ | 5.8 | 18 | 3.8 | Fast release |

**Primary ranking metric**: τ₁/₂ (directly answers "stays oxygenated longest")

**Secondary**: D (intrinsic material property, geometry-independent)

**Quality check**: RMSE (if high → Fick's law may not apply → flag for further investigation)

---

## 8. Summary of Key Equations

| Quantity | Equation | Number |
|---|---|---|
| Fick's Law | ∂C/∂t = D ∂²C/∂x² | (1) |
| Crank-Nicolson stencil | -r C_{i-1}^{n+1} + (1+2r) C_i^{n+1} - r C_{i+1}^{n+1} = RHS | (4) |
| Mesh ratio | r = D Δt / (2 Δx²) | (3) |
| Objective | J(D) = Σ [C_sim - C_data]² | (9) |
| Optimal D | D* = argmin J(D) | (10) |
| Retention half-life | τ₁/₂: time when peak excess halves | (13) |

---

*The Python script `diffusion_D.py` implements all of the above.*
