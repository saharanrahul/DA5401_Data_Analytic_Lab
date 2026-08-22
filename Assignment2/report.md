# Assignment 2 - Results 

## Problem 1: Regression

CV RMSE (mean): 852.96
CV RMSE (std): 47.97
Test MAE: 663.07
Test RMSE: 866.71
Test MAPE: 0.0859(9.59%)
Test R2: 0.836

**Primary metric: MAE**

MAE directly tells budgeting teams the average dollar error per job. which is easier to act on than  a squared-erro metric.
RMSE penalize a few large misses more heavily, but for job-level cost estimation, consistent average error matters more than beign sensitive to rare outliers.


## Problem 2: Classification

Selected threshold: 0.11
validation precision: 0.3235
validation recall: 0.8684
Test TN: 191,  FP:  72, FN: 7, TP: 30
Test accuracy: 0.7367
Test precision: 0.2941
Test recall: 0.8108
Test Specificity: 0.7262
Test F1: 0.4317
Test ROC_AUC: 0.8564

**Justification**
Accuracy alone is misleading here becasue only 12.5% of machines actually fails - a modle predicting"no failure" every time would still score  ~87% accuracy while catching zero real failures. The selected threshold (0.11) is low on purpose, Since the buisness requirement is recall >_ 0.85 on validation; a low threshoold flags moremachines as at-rsik, trading precision for fewer missed failure. On  the test set, recall dropped slightly to  0.81 ( from 0.87 on validation), which is expected since the threshold was tuned on a different sampe - this is reasonably closes and acceptable generalization gap, not a major overfit.