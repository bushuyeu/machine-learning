# Kernel Methods - Questions

Repo: [`bushuyeu/ml-kernel-methods`](https://github.com/bushuyeu/ml-kernel-methods)

---

## Problem 1: Point-to-Plane Distance

> Show that the distance of a point `x` from the plane `w^T x − b = 0` is `(w^T x − b) / ‖w‖`.

1. The distance equals `‖x − proj_P(x)‖`.
2. `x − proj_P(x) = a w` for some scalar `a`, because `w` is normal to `P`.
3. Take the inner product with `w`: `w^T x − w^T proj_P(x) = a ‖w‖²`.
4. `proj_P(x)` lies on `P`, so `w^T proj_P(x) = b`. Combine: `w^T x − b = a ‖w‖²`.
5. Therefore the distance is `‖a w‖ = (w^T x − b) / ‖w‖`. □

**Professor's questions** (probing the geometry behind the proof):

1. Plug step (3) back into step (2) — show explicitly why `a = (w^T x − b) / ‖w‖²`.
2. Does the plane `w^T x − b = 0` contain the origin? When?
   - **Answer:** only when `b = 0` (because `w^T 0 = 0`).
3. When `b ≠ 0`, what is the point on the plane that comes **closest to the origin**? Give an explicit formula.
   - **Hint:** travel from the origin in the direction of `w` (the normal) until you hit the plane.
   - **Expected answer:** `x* = (b / ‖w‖²) · w`. Distance from origin: `|b| / ‖w‖`, which is exactly the formula evaluated at `x = 0`.
4. Now set `b = 0`. The set `S = {x : w^T x = 0}` is a plane through the origin — what is its **dimension**?
5. Why can we talk about "the dimension" of `S = {x : w^T x = 0}` but **not** of `{x : w^T x = b}` when `b ≠ 0`?
   - **Answer:** `S` is a linear subspace (closed under linear combinations); `{w^T x = b}` is an affine set, not a vector space, so dimension-as-a-vector-space is undefined.
6. Definition check: how many linearly independent vectors can you find in `S`? (this _is_ the definition of dimension)
7. If `w, x ∈ R^3`: what is `S`? (a 2-dimensional plane)
8. If `w, x ∈ R^2`: what is `S`? (a 1-dimensional line)
9. In general, in `R^n`: `dim(S) = n − 1`.

**Submission follow-up**: add the explicit closest-point-to-origin formula `x* = (b / ‖w‖²) · w` to the writeup as a corollary of the main result.

---

## Problem 2: Cross-Validated Generalization Bound (`k/n`)

> For an SVC with `k` support vectors trained on `n` points, show that the leave-one-out CV estimate of the generalization error is at most `k/n`.

1. LOO-CV: for each `i`, train `M_i` on the data minus `x_i`, predict on `x_i`. The estimate is `n_wrong / n`.
2. By the representer theorem, if `x_i` is **not** a support vector of the original SVC, then `M_i` has the same boundary, so it correctly classifies `x_i`.
3. If `x_i` **is** a support vector, we don't know whether `M_i` will get it right.
4. Worst case: `n_wrong = k`. In general, `n_wrong ≤ k`. So `n_wrong / n ≤ k / n`. □

**Professor's questions** (extending to leave-two-out):

1. Restate in your own words why LOO-CV gives the `k/n` bound.
2. Now suppose the validation set has size **2** (training set size `n − 2`). What is the bound on the generalization error in this case?
3. Walk through the argument explicitly — don't just give the answer.
4. There are **three cases** for what the validation pair `(x_i, x_j)` contains:
   - both are support vectors,
   - exactly one is a support vector,
   - neither is a support vector.
     How do you handle each case?
5. If the validation pair contains a support vector, do you actually **know** the LOO model will misclassify it?
   - **Answer:** No — even when you remove a support vector, the new model can get lucky and still classify it correctly. So the inequality is strict-or-equal `≤`, **not** equality.
6. Why is the leave-two-out validation-error formula only `≈` and not `=`?
   - **Answer:** because of the randomization over which pair you leave out — the empirical average over pairs is an estimator of the expected error.
7. The per-pair validation error takes values in `{0, 1/2, 1}`. Give the explicit calculation for the probability of each (this is a sum of binomial-style terms over the three cases).

**Submission follow-up**: write out the leave-two-out bound case-by-case, with the probability calculation in question 7.

---

## Problem 3: 20 Newsgroups with Linear / RBF / Matérn Kernels

|                                        | Linear      | RBF                  | Matérn                                          |
| -------------------------------------- | ----------- | -------------------- | ----------------------------------------------- |
| Best hyperparams                       | `C ≈ 11.29` | `C = 5.0`, `γ = 0.9` | `C = 113.14` (rescaled), `kernel='precomputed'` |
| Train accuracy                         | 99.89%      | 99.92%               | **99.97%**                                      |
| Test accuracy                          | **81.59%**  | **82.99%**           | 82.93%                                          |
| `#SVs` (0th binary classifier)         | **1096**    | 2019                 | 1538                                            |
| `#SVs on margin` (0th)                 | 1068 / 1096 | 1993 / 2019          | 1537 / 1538                                     |
| `#misclassified train` (across all 20) | 13          | 9                    | **3**                                           |
| `k/n` for 0th classifier               | **0.097**   | 0.178                | 0.136                                           |
| Test error / 0th classifier            | 0.019       | 0.022                | 0.020                                           |

- Dataset: `fetch_20newsgroups_vectorized` → `n_train = 11314`, `n_test = 7532`, `n_features = 130107`. So **`n < p`** (specifically `11314 ≪ 130107`).
- `OneVsRestClassifier(SVC(...))` for all three kernels; `GridSearchCV` over `C` (linear), `(C, γ)` (RBF), `(C, ν, length_scale)` (Matérn).
- Matérn implemented via a `Pipeline` with a custom precomputed-kernel transformer (`MaternKernelComputer`) that builds the Gram matrix in parallel `200×200` blocks. Hyperparameter search done on a random 1000-sample subset (≈ a few hours), then `C` rescaled by `n_samples / 1000` for the full-data fit.
- Linear-kernel training time: **3932.50 s ≈ 65 minutes**.

**Professor's questions:**

### Comparing kernel results

1. Show me the linear result. Now show me the Matérn (and RBF). They're roughly the same — right? (Test: **81.59 / 82.99 / 82.93**.)
2. Can you really say Matérn is _higher_ than linear?
   - **Answer:** No — these numbers come from a randomized procedure and the spread (~1.4 percentage points) is well below the noise floor. For practical purposes, all three are tied.
3. With the Matérn kernel, did you have trouble training? How long did it take?
   - **Our answer:** ~3 hours per hyperparameter combination on the full dataset, which is why we did `GridSearchCV` on a 1000-sample subset and then rescaled `C` for the final full-data fit.

### Why don't more expressive kernels help?

4. Why doesn't test accuracy improve as you go linear → RBF → Matérn?
5. Our notebook framed this as **overfitting** (`>99.8%` train, `~82%` test) and proposed PCA. **The professor's framing is different**: it's a _structural property of the data_, not a model-capacity problem. Why does the data tend to be **linearly separable** in this regime?
6. How many features (`p`) and how many training examples (`n`) are there? Which is bigger?
   - **Notebook bug to fix**: the markdown cell after the predictions table literally says `i.e. $n\gg p$`, which is the opposite of reality. With `n = 11314` and `p = 130107`, it should be `n ≪ p`.
7. What is the **rank** of the data matrix? (At most `min(n, p) = 11314`.)
8. Why does few points in high-dimensional space imply (almost) linear separability?
   - **Sketch:** `n = 11314` random points in `R^130107` lie on at most an 11314-dimensional affine subspace. In any embedding where they're in general position, you can find a hyperplane separating them under any binary labeling (this is a classical result — see Cover's theorem on the linear separability of random points).
9. **Submission follow-up**: prove this formally (or cite Cover's theorem) and show that `n` random points in general position in `R^p` with `p ≥ n − 1` are almost surely linearly separable for _any_ binary labeling.

### Model selection

10. Looking at the three kernels with their (basically equivalent) accuracies, which one would you choose, and why?
11. **Professor's answer:** **linear**. Reasoning: more support vectors → more "artificial" fit → looser `k/n` bound. Even though linear has the slightly _lowest_ test accuracy (81.59% vs 82.99% / 82.93%), it has the **fewest** support vectors (`1096` vs `2019` / `1538`) and therefore the **tightest** generalization bound from Problem 2 (`k/n = 0.097` vs `0.178` / `0.136`).
12. Connect this back to Problem 2: the `k/n` bound is **actually useful** here — and it's the linear kernel that comes out on top by that criterion.

---

## Problem 4: Credit Risk Prediction

The questions below are the **design constraints** we need to satisfy when we actually write the code, distilled from what the professor told the other groups.

### Asymmetric margins (`### Asymmetric Margins`)

1. The two classes are **not symmetric** — misclassifying a "bad" credit applicant as "good" is much worse than the reverse. What approach gives different margins to the two classes? Justify why it should work.
   - **Practical answer:** use `class_weight={0: w0, 1: w1}` in `SVC` to scale the slack penalties asymmetrically. This effectively shifts the decision boundary.
2. Accuracy is **not** the right metric here. What should we look at instead?
   - **Answer:** precision and recall (and the trade-off between them), and ultimately the **area under the curve**.
3. F1 (harmonic mean of precision/recall) is OK, but **AUC is a much better summary** because it integrates over all decision thresholds, not just one.
4. What curve do we plot? (TPR vs FPR for ROC AUC, or precision vs recall for PR AUC. For class-imbalanced credit data, **PR AUC is usually more informative**.)

### Probability modeling (`### Probability Modeling`)

5. Calibrate probabilities **before** computing recall / AUC. Use the same approach as Problem 3.4: **Platt scaling** via `sklearn.calibration.CalibratedClassifierCV(method='sigmoid')`.
6. Why is Platt scaling reasonable?
   - The sigmoid `Pr(y=1 | x) = 1 / (1 + exp(A f(x) + B))` is monotone increasing in the SVM score `f(x)`, has the right limits (`→ 1` as `f → ∞`, `→ 0` as `f → −∞`), and is bounded in `(0, 1)`.
   - It only fits **two scalars** (`A`, `B`) on a held-out validation set, so the overfitting risk is minimal.

### Ensemble methods (`### Ensemble Methods`)

7. The prompt allows two flavors of base learner:
   - **(i)** randomly training on a subset of the **training data** → use with **boosting** (since bagging already does row resampling internally).
   - **(ii)** randomly training on a subset of the **features** → use with **bagging**.
8. The professor flagged a problem with naïve bagging-of-SVMs: if every base estimator sees the same features, they all fit essentially the same hyperplane and the ensemble adds no variance reduction. **Therefore the right design is bagging with random feature subsets**, not just random row subsets.
9. **Implementation sketch** (from the professor's verbal description): **subclass `SVC`** so that on `fit`, the wrapper:
   - samples a random subset of feature indices,
   - stores them on the instance (e.g. `self.feat_idx_`),
   - calls the parent `fit` on `X[:, self.feat_idx_]`,
   - and on `predict`, projects new `X` onto the same subset before delegating.
     Then pass that wrapper as the `base_estimator` (or `estimator` in newer sklearn) to `BaggingClassifier`.
10. Once implemented, **compare against** a baseline of `BaggingClassifier(SVC(), bootstrap_features=False)` (which only resamples rows). Quantify the difference in test AUC and the variance of base-estimator predictions.
11. Sweep the hyperparameters that matter: **subset size** (number of features per base learner), kernel hyperparameters (`C`, `γ`), and `n_estimators`.

### Dimensionality-reduction trap (raised about another group's solution)

This was _not_ something we did — we explicitly mention PCA in our notebook as a possible remedy but decline to implement it ("for the sake of time"). Including this here as a **lesson learned**, in case we end up adding dim. reduction to Problem 4 later.

12. The other group reduced dimensionality with **truncated SVD / PCA before classification** and saw their accuracy collapse to ~0.54. Why?
    - **Answer:** SVD/PCA picks directions of **maximum variance**, not directions that **separate the classes**. If the class-separating direction happens to be (nearly) orthogonal to the top principal components, projecting onto them destroys the classification signal entirely.
13. So: **never** do PCA/SVD as preprocessing for classification without checking. If we want dimensionality reduction in our credit-risk pipeline, what should we do instead?
    - **Random projections** (Johnson–Lindenstrauss) — preserve pairwise distances in expectation, agnostic to label.
    - **Supervised** dimensionality reduction (LDA, PLS) — explicitly use the labels.
    - **Embedded feature selection** driven by the classifier (e.g. L1-regularized SVM).
14. **Submission follow-up** (only if we actually do dim. reduction in P4): write up **why** PCA/SVD is dangerous before classification, and justify the alternative we choose.

---

## Submission follow-ups (consolidated)

- **P1**: add the explicit closest-point-to-origin formula `x* = (b / ‖w‖²) · w` as a geometric corollary in the writeup.
- **P2**: write out the leave-two-out bound case-by-case, with the probability calculation over `{0, 1/2, 1}`.
- **P3**:
  - **Fix the `n ≫ p` typo** in the notebook markdown — it should be `n ≪ p` (currently `i.e. $n\gg p$`).
  - Re-frame the "overfitting + try PCA" discussion to instead explain that the data is structurally near-linearly-separable because `n ≪ p`.
  - Cite Cover's theorem (or prove it informally) for "few points in high dimensions are almost surely linearly separable".
  - Add a sentence connecting back to Problem 2: linear has the **smallest `k/n`** (`0.097`) and is therefore the model with the tightest generalization guarantee — that's _why_ we should pick it, not "because it's simpler".
- **P4**: actually implement all five subsections. Use `class_weight` (or two-margin formulation) for asymmetry, `CalibratedClassifierCV(method='sigmoid')` for probabilities, and **feature-subset bagging via a custom SVC wrapper** for the ensemble. Report **AUC**, not accuracy. If we add dim. reduction, **don't** use PCA/SVD.
- **For all**: don't _replace_ the original submission. Add a "follow-up" section to the notebook with the corrections so the original work stays visible.
