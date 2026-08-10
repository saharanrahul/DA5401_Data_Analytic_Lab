# DA5401 Data Analytics Lab
## Assignment 1: Linear and Logistic Regression from Scratch

### 1. Objective

Implement linear regression and binary logistic regression from scratch using gradient descent.

You may use only:

- Python standard library
- NumPy
- Pandas

You must not use scikit-learn, statsmodels, TensorFlow, PyTorch, XGBoost, LightGBM, CatBoost, or any other library that directly implements regression, classification, preprocessing, optimisation, or model training.

Your submission must work for any dataset satisfying the input conditions stated below. It must not contain dataset-specific column names, hard-coded numbers, or assumptions based on the public test data.

## 2. Submission

Submit exactly one file named `assignment_submission.py`.

The file must contain every required function with exactly the specified names and arguments.

Do not include code that:

- reads a fixed file from disk;
- requests keyboard input;
- prints intermediate results during import;
- executes training automatically when the file is imported;
- changes global NumPy or Pandas settings.

You may define additional helper functions, but the required functions must remain available at module level.

## 3. General input and output rules

### 3.1 Numerical precision

All calculations must use floating-point arithmetic. Returned numerical arrays must contain finite values.

### 3.2 Target shape

The target `y` may be supplied as a one-dimensional NumPy array, a Pandas Series, or a two-dimensional array with one column. Your functions must internally convert it to a one-dimensional NumPy array.

### 3.3 Reproducibility

Initialise all weights to zero and the intercept to `0.0`. Do not use random initialisation.

### 3.4 Stopping rule

For both models:

1. Compute and store the loss after each parameter update.
2. Stop when either `max_iter` updates have been completed, or the absolute difference between two consecutive stored losses is less than or equal to `tol`.

The returned `loss_history` must contain one value for every completed parameter update.

### 3.5 Invalid inputs

Raise `ValueError` when:

- `X` and `y` contain different numbers of observations;
- `X` contains zero rows or zero columns;
- `learning_rate <= 0`;
- `max_iter` is not a positive integer;
- `tol < 0`;
- any required numerical value is `NaN`, positive infinity, or negative infinity;
- logistic-regression targets contain values other than `0` and `1`;
- the prediction dataset does not contain the same original columns as the logistic-regression training dataset.

Missing values will not be included in the assignment datasets. You do not need to impute them.

# Part A: Linear Regression

## 4. Input assumptions

For linear regression:

- `X` will contain numerical features only.
- `X` may be a Pandas DataFrame or a two-dimensional NumPy array.
- `y` will contain continuous numerical values.
- No missing values, categorical values, or infinite values will be present in valid test inputs.

Use the model:

`y_hat = X @ weights + bias`

and mean squared error:

`MSE = mean((y_hat - y) ** 2)`

## 4.1 Required functions

### `linear_predict(X, weights, bias)`

Returns a one-dimensional NumPy array of predictions.

Required behaviour:

- convert `X` to a two-dimensional floating-point NumPy array;
- compute `X @ weights + bias`;
- return shape `(n_samples,)`.

### `linear_loss(y_true, y_pred)`

Returns mean squared error as a Python `float`.

### `linear_gradients(X, y, weights, bias)`

Returns `dw, db`, where `dw` has shape `(n_features,)` and `db` is a Python `float`.

### `fit_linear_regression(X, y, learning_rate=0.01, max_iter=1000, tol=1e-8)`

Trains the model using full-batch gradient descent.

Return a dictionary with exactly these required keys:

```python
{
    "weights": ...,
    "bias": ...,
    "loss_history": ...
}
```

Requirements:

- `weights`: one-dimensional NumPy array of shape `(n_features,)`;
- `bias`: Python `float`;
- `loss_history`: list of Python floats;
- initialise weights and bias to zero;
- derive, implement, and use the correct gradients for mean squared error;
- do not modify the caller's `X` or `y`.

### `predict_linear_regression(model, X)`

Uses the returned model dictionary and returns a one-dimensional NumPy array.

# Part B: Binary Logistic Regression

## 5. Input assumptions

For logistic regression:

- `X` will be a Pandas DataFrame.
- `X` may contain numerical and non-numerical features.
- A non-numerical feature is any column whose Pandas dtype is not numeric.
- Valid categorical values will be non-missing strings.
- `y` will contain only `0` and `1`.

The implementation must automatically identify all non-numerical columns and one-hot encode them. Every categorical feature must be treated in the same way, irrespective of whether it is nominal or ordinal. No ordinal encoding is to be used.

Use:

`z = X_encoded @ weights + bias`

`p = sigmoid(z)`

and binary cross-entropy:

`BCE = -mean(y * log(p) + (1 - y) * log(1 - p))`

## 5.1 Encoding rules

The encoding behaviour is part of the assignment and will be tested.

For each training DataFrame:

1. Preserve the original column order.
2. Keep every numerical column as one floating-point feature.
3. Treat every non-numerical column as a categorical feature, irrespective of whether it is nominal or ordinal. For each such column:
   - obtain its unique training categories;
   - convert category labels to strings;
   - sort them in ascending lexicographic order;
   - create one binary column for every category using one-hot encoding;
   - do not use ordinal encoding;
   - do not drop a reference category.
4. The encoded feature order must follow original DataFrame column order; within each categorical column, use sorted category order.
5. Store all preprocessing information required to transform future data.
6. During prediction:
   - require the same original column names;
   - internally reorder columns to the training order;
   - use the training categories and encoded feature order;
   - never create a new encoded column.

Prediction datasets used in this assignment will not contain categorical levels that were absent from the corresponding training data.

Example:

Training columns: `age, city, income`

Training categories for `city`: `Chennai, Delhi, Mumbai`

Encoded order: `age, city=Chennai, city=Delhi, city=Mumbai, income`

## 5.2 Required functions

### `fit_preprocessor(X)`

Accepts a Pandas DataFrame and returns a dictionary containing sufficient information to reproduce the encoding exactly.

The dictionary must contain at least:

```python
{
    "original_columns": ...,
    "numerical_columns": ...,
    "categorical_columns": ...,
    "categories": ...,
    "encoded_feature_names": ...
}
```

Required formats:

- `original_columns`: list of strings;
- `numerical_columns`: list of strings;
- `categorical_columns`: list of strings;
- `categories`: dictionary mapping each categorical column to its sorted list of string categories;
- `encoded_feature_names`: list of strings in output order.

Use encoded names in the form `column=category`.

### `transform_features(X, preprocessor)`

Returns a two-dimensional floating-point NumPy array.

Requirements:

- use only metadata from `preprocessor`;
- preserve the exact encoded feature order;
- do not refit categories;
- return shape `(n_samples, n_encoded_features)`.

### `sigmoid(z)`

Returns a NumPy array with the same shape as `z`.

It must be numerically stable. A recommended implementation is:

- for `z >= 0`, use `1 / (1 + exp(-z))`;
- for `z < 0`, use `exp(z) / (1 + exp(z))`.

The function must return finite values for inputs such as `-1000` and `1000`.

### `logistic_loss(y_true, probabilities, epsilon=1e-12)`

Returns binary cross-entropy as a Python `float`.

Before applying logarithms, clip the predicted probabilities so that they remain strictly between `0` and `1`. Use `epsilon` as the lower limit and `1 - epsilon` as the upper limit.

Raise `ValueError` unless `0 < epsilon < 0.5`.

### `logistic_gradients(X_encoded, y, weights, bias)`

Returns `dw, db`, where `dw` has shape `(n_encoded_features,)` and `db` is a Python `float`.

### `fit_logistic_regression(X, y, learning_rate=0.01, max_iter=1000, tol=1e-8, epsilon=1e-12)`

Return a dictionary with exactly these required keys:

```python
{
    "weights": ...,
    "bias": ...,
    "preprocessor": ...,
    "loss_history": ...
}
```

Requirements:

- fit preprocessing using training data only;
- transform the training features;
- initialise weights and bias to zero;
- train using full-batch gradient descent;
- use `epsilon` only for safe loss evaluation;
- derive and implement the correct gradients for binary cross-entropy with the sigmoid model, using the un-clipped sigmoid probabilities;
- ensure every stored loss is finite.

### `predict_proba_logistic(model, X)`

Returns the probability of class `1` as a one-dimensional NumPy array with shape `(n_samples,)`.

### `predict_logistic(model, X, threshold=0.5)`

Returns class predictions as a one-dimensional NumPy integer array containing only `0` and `1`.

Use `1` when probability is greater than or equal to `threshold`; otherwise use `0`.

Raise `ValueError` unless `0 <= threshold <= 1`.

## 6. Assessment through public and private tests

### Public tests

Run:

```bash
pytest -q public_tests.py
```

The public tests check basic compliance and representative functionality. Passing all public tests does not guarantee full marks.

### Private tests

Private tests will use unseen inputs and will assess the same stated requirements, including:

- exact function names and signatures;
- correct formulas;
- correct shapes and return types;
- datasets with different numbers of observations and features;
- convergence on suitable datasets;
- deterministic initialisation and training;
- numerical stability;
- input validation;
- automatic identification and encoding of non-numerical features;
- consistent transformation at prediction time;
- absence of dataset-specific hard-coding.

No private test will require functionality that is not stated in this document.

## 7. Academic integrity and permitted assistance

You may consult course material, NumPy documentation, Pandas documentation and any other source of information.

You are responsible for understanding every line in your submission. Submissions may be inspected for hard-coded outputs, copied implementations, prohibited library usage, or attempts to bypass the test system.

Marks are awarded only for the submitted code that passes the automated tests and satisfies the assignment rules.
