## Problem 1 - Decision Tree Classifier from Scratch
**split_seed**:33

Max_Depth1: Train Accuracy- 0.6319, Validation Accuracy- 0.5849
Max_Depth2: Train Accuracy- 0.6769, Validation Accuracy- 0.5566
Max_Depth3: Train Accuracy- 0.7014, Validation Accuracy- 0.5943
Max_Depth4: Train Accuracy- 0.7526, Validation Accuracy- 0.6226
Max_Depth5: Train Accuracy- 0.7873, Validation Accuracy- 0.6887
Max_Depth6: Train Accuracy- 0.8241, Validation Accuracy- 0.6887

**Underfit_depth:** 1
**overfit_depth:** 6
**best_depth:** 5
**Final test accuracy(Max_depth 5):** 0.6952

**Interpretation:** At depth 1 the tree is too simple to separate the classes, giving the lowest training accuracy and clear underfitting. As depth increases, training accuracy rises steadily and validation accuracy plateaus after depth 5, so at depth 6 the tree keeps fitting training-set noise(train accuracy climbs to 0.8241) without any corresponding gain on validation data - the largest train- validation gap. Depth 5 gives the best validation accuracy while still being the shallower of the two tied-best depths, making it the best generalizing choice, and it holds up well on the untouched test set(0.6952).

## Problem 2 - Random Forest Regression
**Split_seed**:33

Single Tree(baseline):- CV MAE Mean- 11.2241 , CV MAE Std-0.3688
Random Forest(n=10):- CV MAE Mean- 8.0780, CV MAE Std-0.3159
Random Forest(n=50):- CV MAE Mean- 7.6217, CV MAE Std-0.3420
Random Forest(n=100):- CV MAE Mean- 7.5867, CV MAE Std-0.3941
Random Forest(n=200):- CV MAE Mean- 7.5607, CV MAE Std-0.3882

**Selectedn_estimators:**200

**Single-Tree Test Metrics:**
    **MAE:**11.0509
    **RMSE:**13.9260
    **R2:**0.2672

**Selected Random Forest Test Metrics:**
   **MAE:**7.4297
   **RMSE:**9.4748
   **R2:**0.6607


**Interpretation:** The single decision tree overfits the training data and generalizes poorly, giving both the highest test error and the lowest R2 (0.2672). Averaging predictions across 200 randomized trees in the Random Forest substantially reduces variance, cutting test MAE by roughly 33% (11.05 --> 7.43) and nearly tripling R2 (0.267-->0.661). The Random Forsts's cross-validation MAE is also more stable across folds relative to its mean than the single tree's and MAE is also more stable across folds relative to its mean than the single tree's, and MAE continues to improve only marginally past n=50-100 trees, showing diminishing returns from adding more estimators.