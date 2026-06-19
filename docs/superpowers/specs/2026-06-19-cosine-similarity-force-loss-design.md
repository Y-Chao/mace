# Cosine-Similarity Force Loss — Design

**Date:** 2026-06-19
**Status:** Approved (pending spec review)

## Goal

Penalize the **angular** disagreement between DFT (reference) atomic forces and
MACE (predicted) atomic forces during training, via a cosine-similarity term,
controlled by a CLI weight argument. The term can be added on top of either an
**MSE** energy/forces base (`--loss cosine`) or a **Huber** base
(`--loss huber_cosine`). In addition, a force cosine-similarity **metric** is
tracked and reported at every validation, independent of the loss chosen.

## Loss formulation

The cosine term is the same for both bases:

```
cosine_term = mean_i( 1 - cos(F_ref_i, F_pred_i) )
```

- `cos(F_ref_i, F_pred_i)` is the cosine similarity between reference and
  predicted force vectors of atom `i` (each shape `[3]`).
- `1 - cos` lies in `[0, 2]`: `0` when directions align, `2` when anti-parallel.
  Sign matters (anti-parallel maximally penalized), which is physically correct
  for forces.
- Aggregation is a **config-weighted mean** over atoms, reusing MACE's existing
  per-config `weight` / `forces_weight` convention and the DDP-aware
  `reduce_loss` helper.

MSE base (`--loss cosine`):

```
loss = energy_weight*MSE(E) + forces_weight*MSE(F) + cosine_weight*cosine_term
```

Huber base (`--loss huber_cosine`):

```
loss = energy_weight*huber(E) + forces_weight*huber(F) + cosine_weight*cosine_term
```

## Design decisions

1. **Shared helper, thin loss classes.** The cosine term is one loss-agnostic
   helper (`mean_cosine_distance_forces`). Two thin loss classes call it on top of
   their respective base: MSE and Huber. Existing `weighted`, `huber`, and
   `universal` losses are left untouched.
2. **Penalty form:** per-atom `1 - cos`, config-weighted mean. (Not `1 - cos^2`,
   which ignores sign; not a single global cosine over flattened vectors, which
   loses per-atom resolution and weighting.)
3. **Stage Two (SWA) support:** the cosine term persists into Stage Two via a new
   `--swa_cosine_weight` arg and dedicated `get_swa` branches for both
   `cosine` and `huber_cosine`.
4. **Metric vs. loss sign convention.** The *loss* uses `1 - cos` (minimize → 0);
   the *metric* reports `cos` itself (track → 1). Same quantity, conventional sign
   for each context.
5. **Metric is loss-independent.** `cos_sim_f` is computed and reported every
   validation regardless of `--loss`, exactly like force RMSE/MAE.

## Implementation touch points

### 1. Helper function — `mace/modules/loss.py`

Add next to `mean_squared_error_forces` (~line 120):

```python
def mean_cosine_distance_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    )  # [n_atoms]
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    )  # [n_atoms]
    cos_sim = torch.nn.functional.cosine_similarity(
        pred["forces"], ref["forces"], dim=-1
    )  # [n_atoms]; built-in eps guards zero-magnitude forces
    raw_loss = configs_weight * configs_forces_weight * (1.0 - cos_sim)
    return reduce_loss(raw_loss, ddp)
```

Note: weights are 1-D here (no `.unsqueeze(-1)`) because `cosine_similarity(..., dim=-1)`
already reduces the trailing dim to `[n_atoms]`.

### 2. MSE-base loss class — `mace/modules/loss.py`

Modeled on `WeightedEnergyForcesLoss` (~line 246):

```python
class WeightedEnergyForcesCosineLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, cosine_weight=1.0):
        super().__init__()
        self.register_buffer("energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()))
        self.register_buffer("forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()))
        self.register_buffer("cosine_weight",
            torch.tensor(cosine_weight, dtype=torch.get_default_dtype()))

    def forward(self, ref, pred, ddp=None):
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred, ddp)
            + self.forces_weight * mean_squared_error_forces(ref, pred, ddp)
            + self.cosine_weight * mean_cosine_distance_forces(ref, pred, ddp)
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
                f"forces_weight={self.forces_weight:.3f}, "
                f"cosine_weight={self.cosine_weight:.3f})")
```

### 3. Huber-base loss class — `mace/modules/loss.py`

Modeled on `WeightedHuberEnergyForcesStressLoss` (~line 325). Uses Huber for
energy (per-atom-normalized) and forces, then adds the cosine term. Stress is
omitted to keep the class focused on energy+forces+cosine (matching the MSE
variant's scope).

```python
class WeightedHuberEnergyForcesCosineLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0,
                 cosine_weight=1.0, huber_delta=0.01):
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer("energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()))
        self.register_buffer("forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()))
        self.register_buffer("cosine_weight",
            torch.tensor(cosine_weight, dtype=torch.get_default_dtype()))

    def forward(self, ref, pred, ddp=None):
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        reduction = "none" if ddp else "mean"
        loss_energy = torch.nn.functional.huber_loss(
            ref["energy"] / num_atoms, pred["energy"] / num_atoms,
            reduction=reduction, delta=self.huber_delta)
        loss_forces = torch.nn.functional.huber_loss(
            ref["forces"], pred["forces"],
            reduction=reduction, delta=self.huber_delta)
        if ddp:
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = reduce_loss(loss_forces, ddp)
        loss_cosine = mean_cosine_distance_forces(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.cosine_weight * loss_cosine
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
                f"forces_weight={self.forces_weight:.3f}, "
                f"cosine_weight={self.cosine_weight:.3f})")
```

### 4. Exports — `mace/modules/__init__.py`

Add `WeightedEnergyForcesCosineLoss` and `WeightedHuberEnergyForcesCosineLoss`
to the import block (~line 35) and to `__all__` (~line 115).

### 5. CLI args — `mace/tools/arg_parser.py`

- Add `"cosine"` and `"huber_cosine"` to the `--loss` `choices` list (~line 757).
- Add `--cosine_weight` (type `float`, default `1.0`) near `--forces_weight`
  (~line 776).
- Add `--swa_cosine_weight` (type `float`, default `1.0`) near the other
  `swa_*_weight` args (~line 800), with `dest="swa_cosine_weight"`.
- `huber_cosine` reuses the existing `--huber_delta` arg.

### 6. Factory — `mace/tools/scripts_utils.py`

`get_loss_fn` (~line 659), add:

```python
elif args.loss == "cosine":
    loss_fn = modules.WeightedEnergyForcesCosineLoss(
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
        cosine_weight=args.cosine_weight,
    )
elif args.loss == "huber_cosine":
    loss_fn = modules.WeightedHuberEnergyForcesCosineLoss(
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
        cosine_weight=args.cosine_weight,
        huber_delta=args.huber_delta,
    )
```

`get_swa` (~line 742), add branches for both so the cosine term persists in
Stage Two:

```python
elif args.loss == "cosine":
    loss_fn_energy = modules.WeightedEnergyForcesCosineLoss(
        energy_weight=args.swa_energy_weight,
        forces_weight=args.swa_forces_weight,
        cosine_weight=args.swa_cosine_weight,
    )
    logging.info(... cosine weight : {args.swa_cosine_weight} ...)
elif args.loss == "huber_cosine":
    loss_fn_energy = modules.WeightedHuberEnergyForcesCosineLoss(
        energy_weight=args.swa_energy_weight,
        forces_weight=args.swa_forces_weight,
        cosine_weight=args.swa_cosine_weight,
        huber_delta=args.huber_delta,
    )
    logging.info(... cosine weight : {args.swa_cosine_weight} ...)
```

### 7. Tracking metric — `mace/tools/utils.py` + `mace/tools/train.py`

**Metric function** in `utils.py` next to `compute_rmse` (~line 20). Reports the
**mean per-atom cosine similarity** in `[-1, 1]` (1 = perfect alignment):

```python
def compute_cosine_similarity_forces(pred: np.ndarray, ref: np.ndarray) -> float:
    # pred, ref: [n_atoms, 3]
    num = np.sum(pred * ref, axis=-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-9
    return np.mean(num / den).item()
```

**`MACELoss` (`train.py` ~line 589):**

- In `__init__`, add a state to store predicted forces:
  `self.add_state("fps", default=[], dist_reduce_fx="cat")`
  (stored directly rather than reconstructed from `fs - delta_fs`, to avoid
  float drift).
- In `update()` (in the forces block, ~line 636), append:
  `self.fps.append(output["forces"])`
- In `compute()` (forces block, ~line 722), after the existing force metrics:
  ```python
  fps = self.convert(self.fps)
  aux["cos_sim_f"] = compute_cosine_similarity_forces(fps, fs)
  ```

**`valid_err_log` (`train.py` ~line 65):** append `CosSim_F={...:.4f}` to the
printed line for the RMSE/MAE force modes (`PerAtomRMSE`, `TotalRMSE`,
`PerAtomMAE`, `TotalMAE`, and the stress/virials variants). Example:

```
Epoch 10: head: Default, loss=..., RMSE_E_per_atom=... meV, RMSE_F=42.31 meV / A, CosSim_F=0.9876
```

`cos_sim_f` flows into the JSON log automatically via `logger.log(eval_metrics)`.

## Testing

Add unit tests mirroring existing loss tests:

- `mean_cosine_distance_forces` is `0` when `F_pred == F_ref`, positive when
  misaligned, maximal (→ config-weighted `2`) for anti-parallel forces.
- `WeightedEnergyForcesCosineLoss` with `cosine_weight=0` reproduces
  `WeightedEnergyForcesLoss` exactly.
- `WeightedHuberEnergyForcesCosineLoss` with `cosine_weight=0` reproduces the
  energy+forces Huber terms.
- `compute_cosine_similarity_forces` returns `1.0` for identical forces, `-1.0`
  for anti-parallel.
- `__repr__` of both loss classes includes the cosine weight.

## Scale caveat (informational)

Energy/forces terms are MSE/Huber (physical units), while the cosine term is
dimensionless in `[0, 2]`. With the typical default `forces_weight=100`, the
forces term will dominate unless `cosine_weight` is set comparably large.
`cosine_weight` is a tunable knob; this is expected, not a bug.

## Out of scope

- No changes to other existing loss classes (`weighted`, `huber`, `universal`,
  dipole losses).
- No stress/virials term in the cosine loss classes (energy + forces + cosine
  only).
