# Exercise 3: Choose the Right Preprocessing

## Objective

Build a robust and reusable preprocessing pipeline for a housing dataset containing both numerical and categorical features. The goal is to select appropriate preprocessing techniques for different feature types and justify each decision.

---

## Dataset

Use the Housing dataset containing a mixture of:

- Numerical features
- Nominal categorical features
- Ordinal categorical features
- Missing values

---

## Tasks

### 1. Inspect the Dataset

Identify and analyze:

- Numerical features
- Nominal categorical features
- Ordinal categorical features
- Missing values

Perform exploratory analysis to understand the structure and quality of the data before applying preprocessing.

---

### 2. Choose Appropriate Preprocessing Strategies

Select suitable preprocessing techniques for each feature group:

#### Numerical Features
- Missing value imputation
- Feature scaling (if required)

#### Nominal Categorical Features
- Missing value handling
- One-Hot Encoding

#### Ordinal Categorical Features
- Missing value handling
- Ordinal Encoding while preserving category order

---

### 3. Combine Transformations

Use a `ColumnTransformer` to apply different preprocessing steps to different feature groups within a single workflow.

---

### 4. Build a Reusable Pipeline

Construct a preprocessing pipeline using Scikit-Learn's `Pipeline` API that can be:

- Trained on the training dataset
- Applied consistently to validation and test datasets
- Reused in future machine learning workflows

---

## Explainability

For every preprocessing decision, provide a justification explaining why the chosen technique is appropriate for that feature type.

Examples:

- Why use median imputation instead of mean imputation?
- Why use One-Hot Encoding for nominal variables?
- Why use Ordinal Encoding for ordered categories?
- Why scale numerical features?
- Why handle missing values before model training?

---

## Expected Learning Outcomes

By completing this exercise, you will learn how to:

- Identify different feature types in a dataset.
- Handle missing values appropriately.
- Distinguish between nominal and ordinal categorical variables.
- Apply suitable encoding techniques.
- Use `ColumnTransformer` effectively.
- Build reusable preprocessing pipelines.
- Justify preprocessing choices using data-driven reasoning.
- Prepare real-world datasets for machine learning models.

---

## Algorithms and Techniques Used

- SimpleImputer
- OneHotEncoder
- OrdinalEncoder
- StandardScaler / MinMaxScaler (if applicable)
- ColumnTransformer
- Pipeline
- Train / Validation / Test Split

---

## Key Takeaway

Different feature types require different preprocessing strategies. A well-designed preprocessing pipeline ensures consistency, prevents data leakage, and improves the reliability and reproducibility of machine learning models.
