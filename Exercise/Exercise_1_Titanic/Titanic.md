# Exercise 1: Titanic – Build a Preprocessing Workflow

## Objective

Prepare the Titanic passenger dataset for predicting whether a passenger survived.

### Tasks

1. **Inspect the dataset**
   - Examine the dataset shape.
   - Check data types of all features.
   - Review summary statistics.
   - Identify missing values.
   - Distinguish between numerical and categorical features.

2. **Separate the target variable**
   - Extract the `Survived` column as the target variable.
   - Use the remaining columns as input features.

3. **Remove unnecessary features**
   - Identify and remove features that should not be used for prediction (e.g., identifiers or irrelevant attributes).

4. **Split the dataset**
   - Create training, validation, and test datasets.

5. **Handle data preprocessing**
   - Impute missing numerical values.
   - Impute missing categorical values.
   - Encode categorical features appropriately.

6. **Build a preprocessing pipeline**
   - Use `ColumnTransformer` for handling different feature types.
   - Use `Pipeline` to combine preprocessing steps into a single workflow.

---

# Verify the Pipeline

After constructing the preprocessing pipeline:

1. Transform the training and validation datasets.

2. Examine the transformed feature matrix:
   - Are there any remaining missing values?
   - Are all features numerical after preprocessing?
   - What happened to the categorical features?
   - How many features exist after encoding?

3. Add a Logistic Regression classifier to the pipeline.

4. Fit the complete pipeline on the training dataset.

5. Evaluate the model on the validation dataset.

6. Perform a final evaluation on the test dataset.

---

## Expected Learning Outcomes

By completing this exercise, you will learn how to:

- Inspect and understand a real-world dataset.
- Handle missing values effectively.
- Encode categorical variables.
- Build reusable preprocessing pipelines.
- Combine preprocessing and modeling into a single workflow.
- Evaluate machine learning models using proper train, validation, and test splits.
