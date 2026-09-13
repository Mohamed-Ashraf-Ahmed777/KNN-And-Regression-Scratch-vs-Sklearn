# Machine Learning Algorithms Implementation

## 📌 Overview

This project implements and compares fundamental Machine Learning algorithms using two real-world datasets.

The project is divided into two main parts:

1. **K-Nearest Neighbors (KNN) Classification**

   * MAGIC Gamma Telescope Dataset
   * Manual implementation from scratch
   * Scikit-learn implementation
   * Hyperparameter selection
   * Classification evaluation

2. **Linear Regression**

   * California Housing Dataset
   * Direct Solution
   * Ridge Regularization
   * Gradient Descent
   * L1 and L2 Regularization
   * Lasso and Ridge using Scikit-learn
   * Regression evaluation and comparison

The main objective is to understand the mathematical and practical implementation of Machine Learning algorithms rather than relying only on ready-made libraries.

---

# 📂 Project Contents

```text
Machine-Learning-Algorithms/
│
├── KNN/
│   ├── KNN.ipynb
│   └── telescope_data.csv
│
├── Linear-Regression/
│   ├── Linear_Regression.ipynb
│   └── California_Houses.csv
│
├── README.md
└── requirements.txt
```

---

# 1️⃣ K-Nearest Neighbors Classification

## 📊 Dataset

The first part uses the **MAGIC Gamma Telescope Dataset**.

The dataset contains observations describing particle events detected by the MAGIC Gamma Telescope.

The objective is to classify each event into one of two classes:

| Class | Meaning |
| ----- | ------- |
| `g`   | Gamma   |
| `h`   | Hadron  |

Each observation contains 10 numerical features:

```text
fLength
fWidth
fSize
fConc
fConc1
fAsym
fM3Long
fM3Trans
fAlpha
fDist
```

---

## ⚖️ Dataset Balancing

The original dataset contains more Gamma observations than Hadron observations.

To avoid class imbalance, the Gamma class was randomly downsampled to match the number of Hadron observations.

A fixed `random_state=42` was used to make the sampling reproducible.

The resulting dataset contains an equal number of Gamma and Hadron samples.

---

## 🔀 Dataset Splitting

The balanced dataset was divided into three subsets:

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

The training set is used to build the model.

The validation set is used to select the best value of `K`.

The test set is used only for the final evaluation.

---

## 🔧 Feature Scaling

Since KNN depends on distances between data points, feature scaling is important.

Standardization was performed using the training data:

$$
z = \frac{x-\mu}{\sigma}
$$

where:

* $x$ = feature value
* $\mu$ = training-set mean
* $\sigma$ = training-set standard deviation

The mean and standard deviation calculated from the training set were also used to scale the validation and test sets.

---

# 🧠 KNN From Scratch

The KNN classifier was implemented manually without using the KNN classifier provided by Scikit-learn.

The implementation consists of three main steps.

### 1. Calculate Euclidean Distance

The Euclidean distance between two points is:

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

The distance between every test point and every training point is calculated.

### 2. Find the Nearest Neighbors

The calculated distances are sorted from smallest to largest.

The first `K` samples are selected as the nearest neighbors.

### 3. Majority Voting

The classes of the `K` nearest neighbors are collected.

The predicted class is the class with the highest number of votes.

---

## 🔢 K Values

The following values were tested:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
```

The validation set was used to determine the best value of `K`.

---

# 📈 KNN Evaluation

The final model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Metrics

**Accuracy**

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

**Precision**

$$
Precision=\frac{TP}{TP+FP}
$$

**Recall**

$$
Recall=\frac{TP}{TP+FN}
$$

**F1 Score**

$$
F1=2\frac{Precision\times Recall}{Precision+Recall}
$$

---

# 🤖 Scikit-learn KNN

After implementing KNN manually, the same classification task was implemented using:

```python
KNeighborsClassifier
```

The same values of `K` were tested:

```text
1, 3, 5, 7, 9
```

The validation results from the manual implementation were compared with the Scikit-learn implementation.

This comparison helps verify the correctness of the manual implementation.

---

# 📊 KNN Results

## Validation Accuracy

|  K | Manual KNN | Scikit-learn KNN |
| -: | ---------: | ---------------: |
|  1 |        TBD |              TBD |
|  3 |        TBD |              TBD |
|  5 |        TBD |              TBD |
|  7 |        TBD |              TBD |
|  9 |        TBD |              TBD |

**Best Manual K:** TBD

**Best Scikit-learn K:** TBD

**Final Test Accuracy:** TBD

---

# 2️⃣ Linear Regression

## 📊 Dataset

The second part uses the **California Housing Dataset**.

The objective is to predict:

```text
Median_House_Value
```

from the available housing-related features.

---

# 🔀 Dataset Splitting

The dataset was divided into:

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

A fixed random seed was used to make the data selection reproducible.

---

# 🔧 Data Normalization

The input features were standardized using statistics calculated from the training set:

$$
x_{normalized}=\frac{x-\mu_x}{\sigma_x}
$$

The target variable was also normalized:

$$
y_{normalized}=\frac{y-\mu_y}{\sigma_y}
$$

The normalization parameters were calculated from the training set and then applied to the validation and test sets.

---

# ➕ Bias Term

A column of ones was added to the feature matrix.

This allows the model to learn the bias/intercept term as part of the weight vector.

The resulting model can be represented as:

$$
\hat{y}=Xw
$$

---

# 🧮 Linear Regression — Direct Solution

The first regression approach uses the closed-form solution:

$$
w=(X^TX)^{-1}X^Ty
$$

The model predictions were calculated for:

* Training set
* Validation set
* Test set

The performance was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

---

# 🛡️ Ridge Regularization — Direct Solution

To reduce overfitting and improve numerical stability, Ridge regularization was applied.

The regularized solution is:

$$
w=(X^TX+\lambda I)^{-1}X^Ty
$$

The regularization parameter was tested using:

```text
λ = 0.001
λ = 0.01
λ = 0.1
λ = 1
λ = 10
λ = 100
λ = 1000
```

The best value of `λ` was selected according to the lowest validation MSE.

The bias term was excluded from regularization.

---

# 📉 Ridge Regularization Analysis

Training and validation MSE were recorded for each value of `λ`.

A plot was created to visualize the relationship between the regularization parameter and model error.

This allows the effect of regularization on model performance to be analyzed.

---

# ⚙️ Gradient Descent

Linear Regression was also implemented using Gradient Descent instead of the closed-form solution.

The gradient of the Mean Squared Error was calculated as:

$$
\nabla J(w)=\frac{2}{M}X^T(Xw-y)
$$

The weights are updated using:

$$
w:=w-\alpha\nabla J(w)
$$

where:

* $w$ = model weights
* $\alpha$ = learning rate
* $M$ = number of training samples
* $X$ = feature matrix
* $y$ = target values

The implementation tracks training and validation error during the optimization process.

---

# 📉 Gradient Descent Without Regularization

Gradient Descent was first tested without regularization.

The following parameters were used:

```text
Learning Rate = 0.005
Steps = 2000
```

Training and validation errors were recorded during each optimization step.

A plot was created to visualize the decrease in error over time.

---

# 🔴 L1 Regularization — Lasso

L1 regularization was implemented manually using Gradient Descent.

The L1 penalty is:

$$
\lambda\sum_{j}|w_j|
$$

The gradient contribution of the L1 penalty was implemented using the sign of the weights.

The following values were tested:

```text
λ = 0.001
λ = 0.01
λ = 0.1
λ = 1
λ = 10
λ = 100
```

The best value of `λ` was selected based on validation MSE.

The final L1 model was then evaluated on:

* Training set
* Validation set
* Test set

---

# 🔵 L2 Regularization — Ridge

L2 regularization was also implemented manually using Gradient Descent.

The L2 penalty is:

$$
\lambda\sum_{j}w_j^2
$$

The gradient contribution of the L2 penalty was added to the normal gradient during weight updates.

The same set of regularization parameters was tested:

```text
λ = 0.001
λ = 0.01
λ = 0.1
λ = 1
λ = 10
λ = 100
```

The best value of `λ` was selected using validation MSE.

---

# 🧪 Scikit-learn Regression Models

After implementing the regression algorithms manually, equivalent models were implemented using Scikit-learn.

The following models were used:

### Linear Regression

```python
LinearRegression
```

### Lasso Regression

```python
Lasso
```

### Ridge Regression

```python
Ridge
```

Different values of the regularization parameter were tested for Lasso and Ridge.

The validation MSE was used to select the best regularization parameter.

---

# 📊 Regression Evaluation

The regression models were evaluated using:

### Mean Absolute Error

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y_i}|
$$

### Mean Squared Error

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2
$$

Lower values indicate smaller prediction errors.

---

# 📊 Linear Regression Results

| Model                          | Validation MSE | Test MSE | Test MAE |
| ------------------------------ | -------------: | -------: | -------: |
| Direct Solution                |            TBD |      TBD |      TBD |
| Ridge Direct Solution          |            TBD |      TBD |      TBD |
| Gradient Descent               |            TBD |      TBD |      TBD |
| Lasso Gradient Descent         |            TBD |      TBD |      TBD |
| Ridge Gradient Descent         |            TBD |      TBD |      TBD |
| Scikit-learn Linear Regression |            TBD |      TBD |      TBD |
| Scikit-learn Lasso             |            TBD |      TBD |      TBD |
| Scikit-learn Ridge             |            TBD |      TBD |      TBD |

---

# 📈 Visualizations

The project contains several plots for analyzing model behavior.

## KNN

* Validation Accuracy vs. K
* Manual KNN vs. Scikit-learn KNN

## Linear Regression

* Training and Validation MSE vs. Lambda
* Gradient Descent Error vs. Number of Steps
* L1 Training and Validation MSE vs. Lambda
* L2 Training and Validation MSE vs. Lambda
* Scikit-learn L1/L2 Validation MSE vs. Lambda

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# 🎯 Learning Objectives

This project was developed to gain practical understanding of fundamental Machine Learning concepts, including:

* Data preprocessing
* Dataset balancing
* Random data splitting
* Training / Validation / Test sets
* Feature scaling
* Classification
* Regression
* K-Nearest Neighbors
* Euclidean distance
* Majority voting
* Linear Regression
* Closed-form solution
* Matrix operations
* Gradient Descent
* L1 Regularization
* L2 Regularization
* Lasso Regression
* Ridge Regression
* Hyperparameter tuning
* Model evaluation
* Confusion Matrix
* MAE
* MSE
* Comparing algorithms implemented from scratch with Scikit-learn

---

# 👨‍💻 Author

**Mohamed Ashraf Ahmed**

Computer & Communication Engineering Student
Aspiring AI Engineer
