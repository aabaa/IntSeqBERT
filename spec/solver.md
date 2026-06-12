# `src/intseq_bert/solver.py` Implementation Specification

## 1. Overview

`IntegerSolver` reconstructs candidate integers from diagnostic model outputs:

- Magnitude mean and variance,
- Sign prediction,
- Modulo probability distributions.

The solver uses a hybrid algorithm that switches among three search modes according to the width of the inferred integer range. The module also provides `VanillaSolver`, which reconstructs integers from Vanilla Transformer `lm_head` token predictions.

---

## 2. Dependencies

Libraries:

- `math`
- `typing`
- `torch`

Config constants:

| Constant | Value | Purpose |
|----------|-------|---------|
| `MOD_RANGE` | `list(range(2, 102))` | Moduli 2 through 101 |
| `NUM_MODULI` | 100 | Number of moduli |
| `EPSILON` | `1e-6` | Prevent division by zero |

Solver constants:

```python
SOLVER_DENSE_THRESHOLD = 1_000_000
SOLVER_SIEVE_THRESHOLD = 100_000_000_000_000  # 10^14
SOLVER_SIEVE_TARGET = 100_000
SOLVER_BEAM_WIDTH = 10
SOLVER_MAX_ANCHORS = 20

SOLVER_MAG_WEIGHT = 1.0
SOLVER_MOD_WEIGHT = 0.3
```

`SOLVER_MOD_WEIGHT` discounts redundant modulo evidence, such as the overlap among moduli 2, 4, and 8.

Related modules:

- `intseq_models.py` / `vanilla_models.py`: produce solver inputs.
- `base_models.py`: shared model interface.
- `features.py`: reference feature definitions.

---

## 3. Class Design

### `IntegerSolver`

Constructor:

```python
def __init__(
    self,
    mod_range: List[int] = None,
    dense_threshold: int = config.SOLVER_DENSE_THRESHOLD,
    sieve_threshold: int = config.SOLVER_SIEVE_THRESHOLD,
    sieve_target: int = config.SOLVER_SIEVE_TARGET,
    max_anchors: int = config.SOLVER_MAX_ANCHORS,
    beam_width: int = config.SOLVER_BEAM_WIDTH,
):
    ...
```

Attributes:

| Attribute | Description |
|-----------|-------------|
| `mod_range` | List of moduli, defaulting to `config.MOD_RANGE` |
| `dense_threshold` | Mode A to Mode AB threshold |
| `sieve_threshold` | Mode AB to Mode B threshold |
| `sieve_target` | Target candidate count for anchored sieve |
| `max_anchors` | Maximum number of anchor moduli |
| `beam_width` | Beam width for CRT search |

### `solve`

```python
def solve(
    self,
    mag_mu: float,
    mag_log_var: float,
    sign_idx: int,
    mod_log_probs: List[torch.Tensor],
    top_k: int = 5,
) -> List[Dict]:
    ...
```

Arguments:

| Argument | Description |
|----------|-------------|
| `mag_mu` | Predicted Magnitude mean on the `1 + log10(abs(x))` scale |
| `mag_log_var` | Predicted Magnitude log variance |
| `sign_idx` | Sign index: 0=Positive, 1=Negative, 2=Zero |
| `mod_log_probs` | List of log-probability tensors, one per modulus |
| `top_k` | Number of candidates to return |

Return value:

```python
[
    {"value": int, "score": float, "method": "dense" | "sieve" | "crt" | "zero"},
    ...
]
```

Candidates are sorted by descending score. Invalid `sign_idx` values raise `ValueError`.

### `from_model_output`

Static helper that converts a model prediction dictionary into `solve()` arguments.

```python
@staticmethod
def from_model_output(
    predictions: Dict,
    position: int,
    model: "BaseForPreTraining",
    batch_idx: int = 0,
) -> Tuple[float, float, int, List[torch.Tensor]]:
    ...
```

Conversion steps:

1. Extract `mag_mu[batch_idx, position].item()`.
2. Extract `mag_log_var[batch_idx, position].item()`.
3. Compute `argmax` over `sign_logits`.
4. Split `mod_logits` with `_split_mod_logits`.
5. Apply `F.log_softmax(logits, dim=-1)` to each modulus.

---

## 4. Solve Flow

### Step 1: Preprocessing

Sign handling:

- `sign_idx == 2`: immediately return zero candidate.
- `sign_idx == 1`: solve for magnitude and negate final values.
- `sign_idx == 0`: keep positive values.

Search range:

```python
sigma = math.exp(0.5 * mag_log_var)
log10_center = mag_mu - 1
n_min = max(1, math.floor(10 ** (log10_center - 3 * sigma)))
n_max = math.ceil(10 ** (log10_center + 3 * sigma))
width = n_max - n_min
```

Overflow guard:

- For very large `mag_mu`, cap the dense range or route directly to CRT mode.
- Python integers can represent very large candidates, but `10 ** large_float` must be handled carefully.

### Step 2: Mode Dispatch

| Condition | Mode | Method | Description |
|-----------|------|--------|-------------|
| `width <= DENSE_THRESHOLD` | A | `solve_dense()` | Exhaustive search |
| `DENSE_THRESHOLD < width <= SIEVE_THRESHOLD` | AB | `solve_sieve()` | Anchored sieve |
| `width > SIEVE_THRESHOLD` | B | `solve_sparse_crt()` | Sparse CRT |

---

## 5. Scoring

All modes use the same scoring function.

```python
def compute_total_score(
    n: int,
    mag_mu: float,
    sigma: float,
    mod_log_probs: List[Tensor],
    mod_range: List[int],
    mag_weight: float = config.SOLVER_MAG_WEIGHT,
    mod_weight: float = config.SOLVER_MOD_WEIGHT,
) -> float:
    ...
```

Score definition:

```text
score = mag_weight * log_likelihood_magnitude
      + mod_weight * sum(log_likelihood_moduli)
```

Magnitude term:

```python
log10_n = math.log10(n) if n > 0 else 0
mag_target = 1 + log10_n
mag_score = -((mag_target - mag_mu) ** 2) / (2 * (sigma ** 2 + EPSILON))
```

Modulo term:

```python
mod_score = 0.0
for i, m in enumerate(self.mod_range):
    remainder = n % m
    mod_score += mod_log_probs[i][remainder].item()
```

---

## 6. Search Modes

### 6.1 Mode A: Dense Search

Evaluates every integer in `[n_min, n_max]`.

```python
def solve_dense(n_min, n_max, mag_mu, sigma, mod_log_probs, mod_range, top_k):
    candidates = []
    for n in range(n_min, n_max + 1):
        score = compute_total_score(n, mag_mu, sigma, mod_log_probs, mod_range)
        candidates.append({"value": n, "score": score, "method": "dense"})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]
```

Complexity: `O(width * NUM_MODULI)`.

### 6.2 Mode AB: Anchored Sieve

Selects a small set of high-confidence moduli as anchors, uses CRT beam search to generate candidates, then rescales with all moduli.

Anchor selection:

```python
confidences = [(i, mod_log_probs[i].max().item()) for i in range(len(self.mod_range))]
confidences.sort(key=lambda x: x[1], reverse=True)

anchors = []
lcm = 1
for idx, _ in confidences:
    m = self.mod_range[idx]
    lcm = math.lcm(lcm, m)
    anchors.append(idx)

    expected_candidates = (n_max - n_min) // lcm + 1
    if expected_candidates <= self.sieve_target:
        break
    if len(anchors) >= config.SOLVER_MAX_ANCHORS:
        break
```

CRT beam search:

```python
beams = [(0, 1, 0.0)]  # current_x, current_period, cumulative_log_prob
for anchor_idx in anchors:
    m = self.mod_range[anchor_idx]
    top_rems = get_top_remainders(mod_log_probs[anchor_idx], self.beam_width)

    new_beams = []
    for x, M, cum_prob in beams:
        for r, log_p in top_rems:
            try:
                new_x, new_M = solve_crt([(x, M), (r, m)])
                new_beams.append((new_x, new_M, cum_prob + log_p))
            except ValueError:
                continue

    new_beams.sort(key=lambda x: x[2], reverse=True)
    beams = new_beams[:self.beam_width]
```

Candidate enumeration:

```python
candidate_set = set()
for x, M, _ in beams:
    k_start = max(0, (n_min - x + M - 1) // M)
    k_end = (n_max - x) // M
    for k in range(k_start, k_end + 1):
        n = x + k * M
        if n_min <= n <= n_max:
            candidate_set.add(n)
```

### 6.3 Mode B: Sparse CRT

Builds sparse candidates by CRT and evaluates only those candidates. This mode is intended for very wide ranges.

Basis selection:

```python
confidences = [(i, mod_log_probs[i].max().item()) for i in range(len(self.mod_range))]
confidences.sort(key=lambda x: x[1], reverse=True)

basis = []
lcm = 1
width = n_max - n_min
for idx, _ in confidences:
    m = self.mod_range[idx]
    basis.append(idx)
    lcm = math.lcm(lcm, m)
    if lcm > width:
        break
```

Beam search is the same as Mode AB. If a candidate representative `x` is outside the target range, it is shifted by its CRT period `M` into the range when possible.

---

## 7. Mathematical Helpers

### `extended_gcd(a, b)`

Extended Euclidean algorithm. Returns `(g, x, y)` such that `a*x + b*y = gcd(a, b)`.

### `solve_crt(equations)`

Solves a system of congruences.

```python
def solve_crt(equations: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Args:
        equations: [(remainder_1, modulus_1), ...]

    Returns:
        (x, M): smallest non-negative solution x and period M = lcm(m_i)

    Raises:
        ValueError: if the system is inconsistent.
    """
```

The implementation supports non-coprime moduli and raises `ValueError` when residues are contradictory.

### `lcm(a, b)`

Computes the least common multiple. Python 3.9+ also provides `math.lcm()`.

---

## 8. Vanilla Solver

`VanillaSolver` uses `lm_head` logits directly and returns the top-k token predictions that correspond to in-vocabulary integers. It does not use magnitude or modulo diagnostic heads, so out-of-vocabulary integers cannot be reconstructed.

---

## 9. Notes

### 9.1 Numerical Overflow

- Periods `M` and candidates `n` may be extremely large Python integers.
- Guard `math.log10(n)` with `n > 0`.
- Avoid converting huge integers to floating-point values unless bounded.

### 9.2 Log Probabilities

- Probability products underflow, so scoring must sum log probabilities.
- `mod_log_probs` must already be log-softmax outputs.
- Clamp very small log probabilities if `-inf` appears, for example `torch.clamp(log_probs, min=-100)`.

### 9.3 Beam Search

- Trying all CRT combinations is exponential.
- Keep only the top `BEAM_WIDTH` paths.
- Default beam width is `config.SOLVER_BEAM_WIDTH = 10`.

### 9.4 Inconsistent CRT Systems

Contradictory residues, such as `x == 0 (mod 4)` and `x == 3 (mod 2)`, have no solution. Such beams are skipped.

### 9.5 Magnitude Scale

- Model output `mag_mu` is on the `1 + log10(abs(x))` scale.
- Range computation uses `log10(abs(x)) = mag_mu - 1`.
- Scoring compares against `1 + log10(n)`.

---

## 10. Future Extensions

| Item | Description | Priority |
|------|-------------|----------|
| Batch solving | Solve multiple positions in parallel | Medium |
| GPU acceleration | Vectorize dense search | Low |
| Caching | Memoize LCM / CRT results | Low |
