# hyperspectral-svm
# Spectral Kernel Classification: Pavia University & Indian Pines

Exploring kernel-based classification (RBF vs. polynomial SVM) on hyperspectral
imagery, motivated by Dr. Zhang's Spectral Kernel Machines framework.

## Datasets
- **Pavia University**: urban aerial hyperspectral scene, 9 land-cover classes.
- **Indian Pines**: agricultural hyperspectral scene, 16 land-cover classes,
  severe class imbalance (5–2455 samples per class).

## Pipeline
1. Load raw hyperspectral cube + ground truth.
2. Mask out unlabeled/background pixels (applied identically to features and labels).
3. Standardize features (zero mean, unit variance) — fit on train, applied to test.
4. Train/test split, stratified, fixed `random_state`.
5. Baseline RBF SVM, then kernel comparison (RBF vs. polynomial) and hyperparameter
   tuning via GridSearchCV (C, gamma), evaluated on held-out test data.

## Key Findings

### Pavia University
- Baseline RBF (no PCA): **91.87%** accuracy.
- A 3-component PCA (99% variance retained) baseline performed worse (**79.65%**),
  with near-total failure distinguishing Asphalt vs. Bitumen (0/332 correct).
  Root cause: PCA optimizes for variance, not class separability — the subtle
  spectral difference between these chemically similar materials sits in a
  low-variance direction PCA discards.
- Tuned RBF (C=10, gamma=0.1, no PCA): **96.21%** accuracy; Bitumen recall rose
  to 90% (298/332).
- Polynomial kernel (degree 3, no PCA) underperformed RBF (82.17%) and also
  failed completely on Bitumen — confirming the issue is feature representation
  and kernel flexibility, not just PCA.

### Indian Pines
- Baseline RBF (no PCA): **81.27%** accuracy, but 3 of 16 classes (Alfalfa,
  Grass-pasture-mowed, Oats — all under 30 total samples) scored 0.00 F1.
- Initial hypothesis: irrecoverable due to sample scarcity.
- Tuned RBF (C=10, gamma=0.01): **91.34%** accuracy — and all three previously-
  failing classes were largely recovered (e.g. Alfalfa 9/11 correct). This
  **overturned the initial hypothesis**: the failure was driven by
  under-regularized default hyperparameters letting majority classes dominate
  the decision boundary, not an inherent data limitation.

## Lessons on Methodology
- Every reported number here was validated against a correctly matched
  train/test pipeline; three real bugs (label/feature mask misalignment,
  PCA-vs-baseline feature mismatch, train/test scaling mismatch) were found
  and fixed during this work — see commit history.
- `accuracy_score` alone hid meaningful per-class failure in both datasets;
  confusion matrices and macro-averaged F1 were necessary to see it.

## Files
- `pavia_university_svm.ipynb`
- `indian_pines_svm.ipynb`
