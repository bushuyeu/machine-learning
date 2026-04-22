# Notebook Analysis: `kernel_methods_1-4.ipynb`

**Source:** `assigments/ml-kernel-methods/kernel_methods_1-4.ipynb`
**Assignment:** ECE-645-001 Kernel Methods (SP26), Problems 1–4
**Analysis date:** 2026-04-07
**Method:** Three parallel agents reviewing (a) structure & math, (b) code correctness, (c) experiments & results.

---

## 1. Overall Structure

80 cells total (45 markdown, 35 code):

- **Cells 0–2** — title, group members, professor's framing.
- **Cells 3–4** — Problem 1 (point-to-plane distance, proof only).
- **Cells 5–6** — Problem 2 (LOOCV error bound `k/n`, proof only).
- **Cells 7–57** — Problem 3 (20 Newsgroups, linear/RBF/Matérn SVC comparison).
- **Cells 58–79** — Problem 4 (UCI Credit Card Default, SVC + calibration + ensembles).

Problems 1 and 2 are theory. Problems 3 and 4 are applied and carry the bulk of the work (and the bulk of the issues).

---

## 2. Problem 1 — Point-to-Plane Distance

**Task.** Show `d(x) = (wᵀx − b) / ‖w‖`.

**Assessment.** Proof is correct, complete, and well presented. Uses projection of `x − proj_P(x) = a·w`, derives `a‖w‖² = wᵀx − b`, concludes with the signed distance and correctly notes the sign indicates side of the plane.

**No issues.**

---

## 3. Problem 2 — LOOCV Generalization Bound

**Task.** For an SVC with `k` support vectors trained on `n` points, show LOOCV error ≤ `k/n`.

**Assessment.** Proof is tight. Correctly invokes the representer-style argument that removing a non-support vector leaves the classifier unchanged, so only the `k` support vectors can possibly flip. Correctly frames `k/n` as an **upper bound**, not an exact estimate — important distinction.

Minor improvement: the implicit appeal to the representer theorem would be clearer if cited explicitly.

**No correctness issues.**

---

## 4. Problem 3 — 20 Newsgroups (Linear / RBF / Matérn)

### 4.1 Dataset

- 11,314 train / 7,532 test, **130,107 TF-IDF features**. Severe `p ≫ n`.
- Appropriate benchmark. The high-dimensionality is explicitly used later to justify overfitting via Cover's theorem (Cell 27) — good.

### 4.2 Experimental Setup

- Linear SVC: `C ∈ [10⁻⁴, 10⁴]` log-spaced (20 values).
- RBF SVC: `C ∈ {0.2, 1.0, 5.0}`, `γ ∈ {0.3, 0.9, 2.7}` (9 combos).
- Matérn (precomputed kernel): grid-searched **on a 1,000-point subset only**, then parameters rescaled for the full dataset.
- 5-fold CV via `GridSearchCV`, `n_jobs=-1`.

**Gaps:**

- Hyperparameter ranges for RBF/Matérn are not justified. Why those `γ` values?
- No held-out validation set separate from CV.
- **`RANDOM_SUBSET_SIZE = 1000` is not random** — Cell 18 takes `X_train[:1000]` instead of sampling. The variable name is misleading (the author seems to acknowledge it with an asterisk in the comment).
- No global seed is set for Problem 3 (contrast with `SEED = 42` in Problem 4). CV fold determinism is not fully controlled.

### 4.3 Results

| Kernel | Train Acc | Test Acc | #SV | #Margin SV |
|--------|-----------|----------|-----|------------|
| Linear | 99.89% | 81.59% | 1,096 | 1,068 (97.4%) |
| RBF    | 99.92% | 82.99% | 2,019 | 1,993 (98.7%) |
| Matérn | 99.97% | 82.93% | 1,538 | 1,534 (99.7%) |

Extreme train-test gap (~18 pp) is correctly attributed to `p ≫ n`. Only **13 total misclassifications on training** across the three kernels — the models are memorizing.

### 4.4 Problem 2 Bound Verification (Cells 50–54)

Correct arithmetic. Bounds are satisfied but very loose (empirical error ≪ `k/n`). Cell 48 acknowledges the theorem is for binary classifiers while this is 20-class One-vs-Rest, but doesn't rigorously address implications.

### 4.5 Critical Bugs in Problem 3

**🔴 HIGH — Cell 37: copy-paste bug in kernel-intersection analysis.**

```python
supvecs_rbf_matern = np.intersect1d(supvecs_linear, supvecs_matern)  # wrong
```

Should be `np.intersect1d(supvecs_rbf, supvecs_matern)`. The printed "RBF & Matern common support vectors" number is actually the **linear ∩ Matérn** count. Any conclusion about RBF-vs-Matérn similarity is wrong.

**🔴 HIGH — Cell 19: ad-hoc scaling of `C` after subset search.**

```python
opt_params_matern['svm__estimator__C'] *= (len(y_train) / RANDOM_SUBSET_SIZE)
```

This multiplies the best `C` from the 1K-subset search by ~11.3. `C` is not a hyperparameter that scales linearly with `n` — the regularization trade-off is dataset-size independent. The final Matérn model is therefore trained with an unjustified regularization strength and is not directly comparable to Linear/RBF.

**🟡 MEDIUM — Cell 43: inconsistent sparse handling.**

Cell 40 (linear) and Cell 42 (RBF) use `dual_coef_.toarray()[0]`; Cell 43 (Matérn) uses `dual_coef_[0]` directly. Fragile across sklearn versions and paths (especially for the precomputed-kernel path). May crash or silently return the wrong shape.

**🟡 MEDIUM — Cell 26: manual accuracy loop.**

```python
n_correct = sum(a == b for a,b in zip(pred, y))
```

Should be `np.mean(pred == y)` or `accuracy_score()`. Already imported elsewhere in the notebook.

**Refactor opportunity.** The margin-vector extraction logic is duplicated identically in Cells 40/42/43 — extract into one helper. That would also have prevented the Cell 43 sparse inconsistency.

### 4.6 Missing from Problem 3

- No PCA / dimensionality reduction experiment, even though Cell 27 explicitly names it as the solution.
- No per-class accuracy — with 20 classes some are clearly harder than others.
- No ablation holding `C` fixed across kernels for a clean comparison.
- No learning curves, no train-vs-CV curves.
- Why does RBF need ~2× as many SVs as Linear for comparable test accuracy? Not discussed.

---

## 5. Problem 4 — UCI Credit Card Default

### 5.1 Dataset & Splits

30,000 samples, 23 features, binary target (~22% positive). 60/20/20 stratified train/val/test, `StandardScaler` fit **only** on train → no leakage. This is clean and appropriate.

### 5.2 Base SVC

- RBF, GridSearchCV over `C ∈ {0.1, 1, 10, 100}`, `γ ∈ {'scale', 0.001, 0.01, 0.1}`.
- Best: `C=100, γ=0.001`. CV acc 82.06%, validation acc 81.80%.
- 7,471 / 18,000 (41.5%) are support vectors. Dual coefs saturate at `±C`, so the margin constraint is active.

### 5.3 Class Imbalance Problem

```
Default class (1): precision 0.68, recall 0.33
```

**The base model misses two-thirds of actual defaults.** For credit risk this is the whole game, and the notebook correctly identifies that false negatives are costly (Cell 74).

### 5.4 Asymmetric Margins (Cell 75) — 🟡 RESULTS MISSING

Code sets `class_weight = {0:1, 1:k}` for `k ∈ {1, 2, 5, 10}`. **The cell has no visible outputs** — no table, no plot, no interpretation. The math in Cell 74 (class-dependent slack penalty) is correct, but empirical validation of the claim is absent from the notebook.

Additional issue in Cell 75: the "equivalent variance / bandwidth" exposition computes

```python
g_val = 1.0 / (X_train_s.shape[1] * float(X_train_s.var()))
```

`X_train_s.var()` collapses to a **single scalar over the full matrix**, not per-feature variance. sklearn's `'scale'` uses `1 / (n_features * X.var())` on standardized data, so post-scaling this happens to be ~`1/n_features`, but the derivation as written is misleading about what `γ='scale'` means in general.

### 5.5 Platt Scaling (Cell 77) — 🟡 RESULTS MISSING + METHODOLOGY

```python
prob_svc = SVC(..., probability=True, random_state=SEED)
prob_svc.fit(X_train_s, y_train)
```

Two issues:

1. The calibration plot setup code exists but **no figure is shown** in the notebook state observed.
2. `probability=True` fits Platt's sigmoid via internal CV on the **training set**. For a cleaner result, calibration should be fit on the validation split (or use `CalibratedClassifierCV` with a held-out fold). As written, the calibration curve will be optimistically biased.

### 5.6 Ensembles (Cells 78–79) — 🔴 NOT IMPLEMENTED

Cell 78 describes bagging and boosting in markdown. **Cell 79 is empty** (or `...`). This is a **required deliverable** of Problem 4 (explicit in the PDF): boosting (e.g. AdaBoost) and bagging, with exploration of subset size and kernel hyperparameters. Neither is present.

### 5.7 Missing from Problem 4

- No test-set numbers anywhere — only CV and validation.
- No ROC/AUC, PR curve, or confusion matrix at different thresholds — all standard for imbalanced binary tasks.
- Cost-sensitive results (Cell 75) — not reported.
- Calibration curve (Cell 77) — not rendered.
- Ensembles (Cell 79) — not implemented.

---

## 6. Reproducibility

| Section | Global seed | CV seed | Deterministic splits |
|---------|-------------|---------|----------------------|
| Problem 3 | ❌ none | partial | yes for train/test |
| Problem 4 | ✅ `SEED=42` | partial (via SVC) | yes |

No assertions on loaded/pickled model shapes, so silent file-corruption failures are possible.

---

## 7. Completeness Scorecard

| Problem | Status | Notes |
|---------|--------|-------|
| 1 | ✅ complete | Rigorous proof. |
| 2 | ✅ complete | Correct bound; nice framing. |
| 3 | ⚠️ complete but buggy | Cell 37 and Cell 19 invalidate part of the analysis. |
| 4 | ❌ incomplete | Ensembles missing; Cell 75 and Cell 77 have no reported results; no test-set evaluation. |

---

## 8. Priority Fix List

**Before submission:**

1. **Cell 37** — fix the RBF∩Matérn intersection and re-run the similarity discussion.
2. **Cell 19** — drop the `C *= n_full/n_subset` scaling; either re-grid on a larger Matérn sample or accept the subset-found `C` as-is and justify.
3. **Cell 43** — normalize sparse handling into a helper (`get_margin_vectors(svm, C)`), use it in cells 40/42/43.
4. **Cell 75** — actually print / plot the class-weight sweep results (F1, recall, precision vs. `k`).
5. **Cell 77** — render the calibration curve; ideally fit Platt scaling on the val split via `CalibratedClassifierCV`.
6. **Cell 79** — implement bagging and boosting. This is a required deliverable.
7. **Problem 4 test set** — report final metrics on the untouched 20% test split.

**Nice to have:**

8. Set a global seed in Problem 3 (Cell 9-ish) for full reproducibility.
9. Add per-class accuracy for the 20-class problem.
10. Add a brief PCA or TruncatedSVD experiment to support the Cell-27 discussion.
11. Fix `X_train_s.var()` exposition in Cell 75 (use `.var(axis=0).mean()` if the point is to match sklearn's `'scale'`).
12. Replace the manual accuracy loop in Cell 26 with `accuracy_score` / `np.mean`.
