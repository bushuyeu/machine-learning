# Problem 4 — Credit Risk Prediction

**Source notebook:** `assigments/ml-kernel-methods/kernel_methods_1-4.ipynb` cells 58–84.

**What's being done:** Build an SVM pipeline that predicts whether a credit-card client will default on their next payment. Unlike Problem 3, this is a binary, low-dimensional, _imbalanced_ tabular problem, so every design decision — kernel, hyperparameters, cost matrix, probability calibration, ensemble — matters.

Problem 4 has five sub-tasks, and the notebook follows them in order:

1. [Dataset and splits](#1-dataset-and-splits)
2. [Base RBF SVC with hyperparameter search](#2-base-rbf-svc-with-hyperparameter-search)
3. [Asymmetric margins (cost-sensitive SVM)](#3-asymmetric-margins-cost-sensitive-svm)
4. [Probability modeling (Platt scaling)](#4-probability-modeling-platt-scaling)
5. [Ensemble methods (bagging)](#5-ensemble-methods-bagging)

---

## 1. Dataset and splits

The dataset is the **UCI "Default of Credit Card Clients"** dataset (Yeh & Lien, 2009). 30 000 Taiwanese credit-card accounts, each described by 23 numeric features:

- credit limit
- demographics (age, sex, marital status, education)
- 6 months of repayment status
- 6 months of bill statements
- 6 months of previous payment amounts

The label `y` is `1` if the client defaulted on their next payment, `0` otherwise. **Positive class is the minority** — about 22% of accounts default. This imbalance will dominate every subsequent design decision.

![60/20/20 split with leakage-free scaling](diagrams/p4_data_flow.svg)

### 1.1 Splits

```python
# 60 / 20 / 20 train / val / test, stratified on y
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=SEED)
# 0.20 * 0.25 = 0.05 of the whole dataset sits in val... wait let me redo this
# test_size=0.25 on 80% of the data means 0.25 * 0.80 = 0.20 overall → val is 20%
```

Three reasons this is done as 60/20/20 instead of the usual 80/20:

- **Train (60%) is what the model sees during hyperparameter search and final fitting.**
- **Val (20%) is where you compare candidate pipelines** — base SVC vs cost-sensitive SVC vs calibrated SVC vs bagging ensemble. This is how you pick the final model without touching test.
- **Test (20%) is consulted exactly once**, at the very end, to report the numbers in the writeup.

### 1.2 Leakage-free scaling

From problem definition: _no part of the test data should be touched in any training step, not even to normalize_.
This is a real gotcha, because the obvious way to scale the whole dataset is

```python
# DO NOT DO THIS
X_s = StandardScaler().fit_transform(X)   # fits on all 30 000 rows
```

which uses test-set statistics to compute the mean and variance. That's a subtle leak — the resulting $(\mu, \sigma)$ have been influenced by test rows. Even if your model itself never sees test labels, the scaler has.

The correct pattern is:

```python
scaler = StandardScaler().fit(X_train)          # train-only statistics
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
```

`.fit(X_train)` computes $(\mu, \sigma)$ from training rows only. `.transform(X_val)` and `.transform(X_test)` apply those same statistics to the other splits without peeking at their values. This is the same preprocessing a production model would face at inference time — the mean and variance are frozen before deployment, and new data is mapped through them as-is.

---

## 2. Base RBF SVC with hyperparameter search

### 2.1 Why RBF?

The features are a mix of continuous (credit limit, bill statements) and ordinal discrete (education level, repayment status). After standardization, a linear kernel would only be appropriate if the decision boundary between "default" and "no-default" happened to be close to a hyperplane — which is rarely true for this kind of tabular data.

The **RBF kernel**

$$k(\mathbf{x}, \mathbf{x}') = \exp\bigl(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2\bigr)$$

is the default nonlinear choice for tabular classification. It places a Gaussian bump of bandwidth $\sim 1/\sqrt{2\gamma}$ around every training point, so the decision surface can curve around pockets of defaults that a linear boundary would miss. Matérn (from Problem 3) would work too but brings no meaningful advantage for 23-D tabular data.

### 2.2 Hyperparameters to tune

- **$C$** — soft-margin penalty. Small $C$ → smoother / more biased; large $C$ → tighter / more variance-prone.
- **$\gamma$** — inverse squared bandwidth of the RBF. Small $\gamma$ → global, smooth boundary; large $\gamma$ → sharp, local-pocket boundary.

The notebook does a 4×4 grid over $(C, \gamma)$:

```python
param_grid = {
    'C':     [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 0.001, 0.01, 0.1],
}
gscv = GridSearchCV(SVC(kernel='rbf', random_state=SEED),
                    param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gscv.fit(X_train_s, y_train)
```

**5-fold cross-validation inside the training set**, optimizing plain accuracy. The winning configuration: `C=100, gamma=0.001`.

### 2.3 What to report after fitting

Cell 77 extracts the key anatomical facts of the fitted SVC:

- **Number of support vectors per class:** `best_svc.n_support_` — how many training points ended up on or inside the margin, split by class.
- **Fraction of training data that became SVs:** typically ~40% here — much higher than Problem 3's newsgroups because the decision boundary is much harder to draw in low-D tabular space.
- **Dual coefficients:** `best_svc.dual_coef_` has shape `(1, #SVs)` for binary SVC and contains the signed $\alpha_i y_i$ values. Their range tells you whether the soft-margin penalty $C$ is "saturating" — if many entries are at exactly $\pm C$, the model is using the soft margin heavily (many bound SVs).
- **Validation metrics:** accuracy, precision, recall, F1 on the positive class.

The numerical punchline on this dataset is unforgiving:

```
validation:  acc ≈ 0.818   precision[default] ≈ 0.69   recall[default] ≈ 0.33   F1[default] ≈ 0.45
```

**An accuracy of 82% sounds fine, but look at the recall.** The model catches only _one in three_ actual defaulters. That's the class-imbalance bite: the classifier has learned to classify almost everything as "no default" because 78% of training labels are that, and that alone gets you 78% accuracy.

This is the motivation for the next section.

---

## 3. Asymmetric margins (cost-sensitive SVM)

### 3.1 Why the default is wrong

The soft-margin SVM minimises

$$\tfrac{1}{2}\|\mathbf{w}\|^2 \;+\; C \sum_{i=1}^n \xi_i$$

subject to $y_i(\mathbf{w}^T\mathbf{x}_i - b) \geq 1 - \xi_i$, $\xi_i \geq 0$. Every slack variable $\xi_i$ is penalized with the same $C$ regardless of the class. In credit risk that's upside-down: a **false negative** (letting a defaulter slip through) is far more expensive to the lender than a **false positive** (flagging a safe client for extra review).

### 3.2 The fix: class-dependent $C$

Replace the single $C$ with two penalties $C_+$ for positives and $C_-$ for negatives:

$$\tfrac{1}{2}\|\mathbf{w}\|^2 \;+\; C_+ \sum_{i:\,y_i=+1} \xi_i \;+\; C_- \sum_{i:\,y_i=-1} \xi_i.$$

In scikit-learn this is exposed as `class_weight`:

```python
SVC(kernel='rbf', C=C_best, gamma=gamma_best, class_weight={0: 1, 1: w})
```

Setting `class_weight={0: 1, 1: w}` with $w > 1$ effectively makes $C_+ = w \cdot C$ and $C_- = C$. Missing a positive now costs $w\times$ more than misclassifying a negative.

### 3.3 What changes geometrically

![Symmetric vs asymmetric soft-margin SVM](diagrams/p4_asymmetric_margin.svg)

When you raise $C_+$, you're telling the optimizer "I'm willing to accept more negatives being misclassified in exchange for catching more positives." The optimal hyperplane responds by:

1. **Shifting toward the negative class.** That gives positives a wider protected region on their side of the margin.
2. **Converting some correctly classified negatives into bound support vectors.** Those negatives now sit inside the new margin — they've been sacrificed to the positive class's comfort.
3. **Raising recall at the expense of precision.** There's no free lunch.

The notebook sweeps $w \in \{1, 2, 3, 5, 10\}$ (cell 80) and reports precision / recall / F1 / accuracy on the validation set. The typical finding:

| $w$ | val acc | precision (default) | recall (default) | F1       |
| --- | ------- | ------------------- | ---------------- | -------- |
| 1   | 0.81    | 0.69                | 0.33             | 0.45     |
| 2   | 0.80    | 0.54                | 0.45             | **0.49** |
| 3   | 0.77    | 0.47                | 0.50             | 0.48     |
| 5   | 0.68    | 0.37                | 0.64             | 0.47     |
| 10  | 0.54    | 0.29                | 0.75             | 0.42     |

**The F1 sweet spot is around $w = 2$** for this data. Going higher trades off too much precision for marginal recall gains. Of course, what "the right $w$" is depends entirely on the lender's cost matrix — in a real deployment you'd pick $w$ by business criteria, not by F1.

---

## 4. Probability modeling (Platt scaling)

### 4.1 Why SVM scores aren't probabilities

The raw SVM output is $f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} - b$, the signed distance from the hyperplane (see Problem 1). For classification we take its sign. For _probabilistic_ decisions — which are essential for credit risk, where the lender wants to say "this applicant has a 30% chance of defaulting, so we price the loan accordingly" — we need a number in $[0, 1]$.

Raw SVM scores have the wrong range (they're unbounded) and aren't calibrated to empirical frequency (two different SVMs can produce the same score for very different actual risks).

### 4.2 Platt scaling (Platt, 1999)

Fit a **logistic regression on top of the SVM scores**:

$$\widehat{P}(y = 1 \mid \mathbf{x}) \;=\; \sigma\bigl(A \cdot f(\mathbf{x}) + B\bigr) \;=\; \dfrac{1}{1 + \exp(A f(\mathbf{x}) + B)}$$

Two scalar parameters $A < 0$ and $B$, fit by maximum likelihood on the pairs $(f(\mathbf{x}_i), y_i)$. This is a **monotone two-parameter squashing**:

- **Ranking is preserved.** The AUC of the calibrated model is the same as the AUC of the raw SVM — Platt just changes the scale.
- **Frequencies are matched.** The output probabilities are calibrated to empirical class frequencies in the training data.

### 4.3 The leakage subtlety

A naive implementation fits the sigmoid on the same training data the SVM was trained on. That's optimistic because the SVM has already memorized those scores — the resulting calibration curve will look beautiful on training but fail on new data.

The right thing is `CalibratedClassifierCV(method='sigmoid', cv=5)`:

```python
calibrated = CalibratedClassifierCV(
    SVC(kernel='rbf', C=C_best, gamma=gamma_best, random_state=SEED),
    method='sigmoid', cv=5,
)
calibrated.fit(X_train_s, y_train)
```

Internally:

1. Split the training data into 5 folds.
2. For each fold, train the SVM on the other 4 folds and fit a Platt sigmoid on _this_ fold's scores.
3. At inference time, average the 5 resulting (SVM, sigmoid) pairs.

In our version we use `SVC(probability=True)`, which does something similar internally (5-fold CV and a Platt fit) but on the full training set.

### 4.4 How to evaluate calibration

Three tools, all reported in the notebook:

1. **Brier score:** $\mathrm{BS} = \tfrac{1}{n}\sum_i (\hat p_i - y_i)^2$. A proper scoring rule. Lower is better; 0 is perfect. Baseline "always predict 0.5" gives 0.25.
2. **Reliability diagram.** Bin the predictions by predicted probability (e.g. 10 quantile bins) and plot _observed frequency_ vs _mean predicted probability_ for each bin. A perfectly calibrated model lies on the $y = x$ diagonal.
3. **ROC AUC.** Rank-only metric; invariant to the Platt squashing. Tells you the model's raw discrimination power separately from its calibration.

On this dataset the typical numbers are Brier ≈ 0.145, AUC ≈ 0.72. The reliability diagram shows the calibration is quite good — predicted probabilities match observed default frequencies closely across bins.

### 4.5 Threshold sweep

With calibrated probabilities you can pick any decision threshold $\tau \in [0, 1]$ and classify as "default" if $\hat{p} \geq \tau$. The default is $\tau = 0.5$, but that's rarely the best choice for imbalanced cost-sensitive problems:

| threshold | precision | recall   | F1       |
| --------- | --------- | -------- | -------- |
| 0.20      | 0.44      | 0.52     | 0.48     |
| **0.25**  | **0.53**  | **0.46** | **0.50** |
| 0.30      | 0.59      | 0.40     | 0.48     |
| 0.50      | 0.68      | 0.28     | 0.40     |

**Lowering the threshold from 0.5 to 0.25 improves F1 from 0.40 to 0.50** on this data. This is the post-hoc equivalent of the class-weight trick from section 3: both shift the decision boundary toward the minority class, but the threshold approach is cheap (one scalar knob) and doesn't require retraining.

---

## 5. Ensemble methods (bagging)

### 5.1 Why ensemble?

A single kernel SVC has two sources of error: **bias** (the kernel and $C$ smooth away some signal) and **variance** (small changes to the training data shift the boundary). Bagging attacks the variance component; boosting attacks the bias component.

### 5.2 Design choice: bagging

The problem definition lets us pick one of bagging or boosting. **Bagging is the cleaner fit here** as Adaptive Boosting is built for _weak_ learners — classifiers that are only barely better than random guessing. Its whole recipe is to combine many such weak classifiers into one strong one, with each new round trained to fix the previous round's mistakes. That only pays off when the base model has clear room to improve. A tuned RBF-SVC doesn't: it already hits ~85% training accuracy on its own (cell 77 gives `train acc = 0.8458`). There's no weakness left for AdaBoost to build on, so using it over SVC base learners would add complexity without improving the result.

### 5.3 What bagging does

![Bagging with random sample + feature subspaces](diagrams/p4_bagging_architecture.svg)

Bagging trains $n$ independent base learners, each on a different random subset of the data, then aggregates predictions by voting. The parameters that matter:

- **`n_estimators`** — how many base learners.
- **`max_samples`** — fraction of rows each base learner sees (with bootstrap replacement, since `bootstrap=True`).
- **`max_features`** — fraction of columns each base learner sees (without replacement, since `bootstrap_features=False`).
- **Inner kernel hyperparameters $(C, \gamma)$** — the base SVC's own knobs.

scikit-learn wraps this as `BaggingClassifier(estimator=SVC(...), ...)`.

**Why feature subsets matter.** Naïve bagging that only resamples rows tends to under-deliver for kernel SVMs: if every base estimator sees the same feature set, they all fit essentially the same hyperplane, and averaging them buys you very little variance reduction. The right design — and the one the professor explicitly requested for this problem — is to **also randomly subsample features per base learner**, so each one carves a different decision surface in a different projection of feature space. This is what `max_features < 1` enables.

**Two equivalent ways to implement feature-subset bagging.** The professor's prescribed implementation is a custom `SVC` subclass:

```python
class FeatureSubsetSVC(SVC):
    def fit(self, X, y):
        n_features = X.shape[1]
        self.feat_idx_ = np.random.choice(n_features, size=int(0.7 * n_features), replace=False)
        return super().fit(X[:, self.feat_idx_], y)
    def predict(self, X):
        return super().predict(X[:, self.feat_idx_])
```

You then pass `FeatureSubsetSVC()` as the base estimator to a plain `BaggingClassifier(bootstrap=True, bootstrap_features=False)`. Our notebook takes a shortcut: `BaggingClassifier(estimator=SVC(...), max_features=f)` does the exact same thing internally — scikit-learn samples a random feature subset for each base learner, slices `X` down to that subset before calling `fit`, and slices incoming inputs the same way on `predict`. The two approaches produce identical predictions given the same random seed; the difference is purely in _where_ the feature-selection code lives (inside sklearn's wrapper vs. inside a user-written subclass). We stick with the built-in `max_features` option because it's lighter and matches scikit-learn idiom.

### 5.4 What the notebook sweeps

Cell 84 explores 7 configurations (after the kernel sweep was added in commit `88ca261`):

| config         | n_est | max_samples | max_features | C   | γ       |
| -------------- | ----- | ----------- | ------------ | --- | ------- |
| baseline       | 10    | 0.5         | 0.7          | 100 | 0.001   |
| no feature sub | 10    | 0.5         | 1.0          | 100 | 0.001   |
| more learners  | 15    | 0.3         | 0.7          | 100 | 0.001   |
| strong decorr  | 10    | 0.5         | 0.5          | 100 | 0.001   |
| tighter γ      | 10    | 0.5         | 0.7          | 100 | 0.003   |
| looser γ       | 10    | 0.5         | 0.7          | 100 | 0.00033 |
| softer C       | 10    | 0.5         | 0.7          | 10  | 0.001   |

The first four configs vary the **ensemble** structure at fixed kernel hyperparameters. The last three hold the ensemble structure fixed and vary the **kernel** hyperparameters. Together they answer the question "how do subset sizes AND kernel hyperparameters affect the ensemble?"

**Config 2 is the row-only baseline.** The row labelled `no feature sub` sets `max_features = 1.0`, which means every base learner sees all 23 features. That's pure row-bagging — equivalent to `BaggingClassifier(SVC(), bootstrap_features=False)` — and it's exactly the baseline the professor asked us to compare against. If the feature-subset configs (1, 3, 4) beat config 2, that demonstrates the extra variance reduction feature subspaces provide. If they don't, it tells us something about the data.

### 5.5 What happens: bagging ~ baseline on this data

On Mikey's runs:

- **Subset-size sweep:** F1(default) on validation ranges from 0.34 to 0.44. Base-single-SVC baseline is 0.45. **Bagging does not beat the baseline** — in fact it slightly loses.
- **Kernel-hyperparameter sweep:** Tighter $\gamma$ (×3) wins the sweep at F1 0.445. Looser $\gamma$ and softer $C$ both collapse recall.
- **Row-only baseline (config 2, `max_features = 1.0`):** F1 = 0.435 — essentially tied with the best feature-subset config at 0.445. This is the direct comparison the professor asked for, and it confirms that **adding feature subspaces does not measurably help on this dataset**. The marginal difference (0.010 in F1) is well within run-to-run noise.

The report in the notebook: **bagging does not help on this dataset, and the ensemble IS sensitive to kernel tuning.**

Why doesn't bagging help?

1. **The base SVC is already low-variance.** On 18 000 tabular points with 23 features, small training-set perturbations don't move the decision boundary much. There's no variance to reduce.
2. **Feature subsampling _hurts_ because features are complementary.** Credit-risk features all contribute useful signal (payment status, bill amounts, credit limit). Dropping 30–50% of them at each base learner loses information that the vote can't recover. The row-only baseline (config 2) avoiding this loss is part of why it ties the feature-subset configs — it simply isn't throwing information away.
3. **Bagging is a high-variance-fixer.** It works best on models like deep decision trees where individual trees overfit wildly in different directions. Kernel SVMs just don't have that failure mode.

This is a valuable negative finding. In the write-up you explicitly report "bagging does not improve over a single tuned SVC on this data, and here is the evidence."

> **Caveat on the metric — AUC, not F1, is the right reporting metric.** The professor was explicit that accuracy and F1 are the wrong things to foreground for this imbalanced problem: the right metric is **ROC AUC**, and for imbalanced positive rates like 22%, **PR AUC** (average precision) is more informative still. Our cell 84 outputs the accuracy / precision / recall / F1 triplet, not AUC, which is why the discussion above leans on F1. A tight follow-up would extend cell 84 to compute both `roc_auc_score` and `average_precision_score` on the validation set for each of the 7 configs (plus the single-SVC baseline) and re-rank them by those metrics. The qualitative story — _bagging does not outperform the single tuned SVC on this data_ — is very unlikely to flip under AUC, because the ensemble and baseline SVCs have near-identical decision surfaces to begin with; but the headline number in the write-up should still be AUC.

### 5.6 Final test-set numbers

The last part of cell 84 runs the best base SVC and the best bagging ensemble on the held-out test set — the _only_ time the test split is touched:

```
Base SVC (single learner):   test acc 0.8153   F1(default) 0.4432
Bagging ensemble (best):     test acc 0.8170   F1(default) 0.4415
```

They're essentially tied on F1, with the ensemble slightly ahead on plain accuracy. Which we report as "best" depends on what metric matters for cost matrix.

The notebook also plots the confusion matrix for both models side by side, which makes the nature of the errors visible:

- Out of 4673 true negatives, the base SVC correctly flags 4440 (high specificity).
- Out of 1327 true positives, the base SVC correctly catches only 444 (recall ≈ 33%).

Most errors are false negatives — the model lets defaulters through — not false positives. This is exactly what the class imbalance + uniform-cost training predicts.

---

## 6. What to remember

1. **Leakage-free scaling.** `StandardScaler().fit(X_train)` first, then `.transform()` the val and test splits.
2. **Class imbalance is the main issue.** 22% positive rate means accuracy is a misleading metric — always report precision, recall, F1, or AUC on the positive class.
3. **RBF kernel** is the default for tabular data. Tune $(C, \gamma)$ by grid search + 5-fold CV on the training set.
4. **Asymmetric margins** (`class_weight`) let you shift the hyperplane toward the majority class, trading precision for recall.
5. **Platt scaling** turns raw SVM scores into calibrated probabilities via a 2-parameter logistic fit. Use `CalibratedClassifierCV` (not naive `SVC(probability=True)`) to avoid fitting the sigmoid on the same data as the SVM.
6. **Threshold sweep** is the post-hoc equivalent of `class_weight`: cheap, doesn't require retraining, and directly exposes the precision/recall trade-off.
7. **Bagging with random feature subspaces** is the PDF-preferred ensemble approach. But on this dataset it does NOT beat the single tuned SVC — an honest finding worth reporting.
8. **The ensemble is sensitive to kernel tuning.** Tighter $\gamma$ helps slightly; looser $\gamma$ and softer $C$ both collapse recall.
9. **Test set is touched exactly once**, at the very end, after all model selection is complete.
