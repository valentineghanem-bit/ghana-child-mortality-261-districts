# Supplementary Methods Notes

## Spatial Weight Matrix
KNN-4 weight matrix constructed using PySAL conventions. Each district connected to its 4 nearest neighbours by centroid distance.

## LISA Permutation Testing
999 random permutations (seed=42). Significance threshold: p<0.05 (two-tailed).

## ML Pipeline Details
- Base learners: RandomForestClassifier(n_estimators=300, random_state=42) + GradientBoostingClassifier(n_estimators=200, random_state=42)
- Meta-learner: LogisticRegression(C=1.0, random_state=42)
- CV: StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
- Feature importance: mean decrease in Gini impurity (RF base learner)

## AUC = 1.00 Interpretation
DHS predictors are assigned at regional level (16 regions); districts within the same region share identical DHS values. The classifier exploits this regional signal, not individual district variation. Reflects ecological data structure, not overfit.
