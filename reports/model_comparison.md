| model | imbalance_strategy | roc_auc | pr_auc | ks | precision | recall | f1 | train_seconds |
|---|---|---|---|---|---|---|---|---|
| **xgboost_weighted** (best) | scale_pos_weight | 0.720 | 0.391 | 0.319 | 0.324 | 0.665 | 0.435 | 7.800 |
| xgboost | none | 0.720 | 0.390 | 0.320 | 0.563 | 0.088 | 0.152 | 8.200 |
| xgboost_smote | smote | 0.712 | 0.380 | 0.309 | 0.525 | 0.111 | 0.183 | 60.100 |
| random_forest | class_weight | 0.709 | 0.377 | 0.301 | 0.314 | 0.660 | 0.426 | 22.700 |
| logistic_regression | class_weight | 0.708 | 0.370 | 0.303 | 0.313 | 0.666 | 0.426 | 6.000 |
