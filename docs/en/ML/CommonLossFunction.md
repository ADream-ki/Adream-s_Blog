---
title: Twelve Common Loss Functions
description: In machine learning, a loss function is a metric that measures the difference between the predicted values of a model and the true values, and it is used to guide the training process of the model.
author: Adream
cover: false
hiddenCover: false
hidden: false
readingTime: true
comment: false
date: 2021-06-12
tags: 
  - Algorithm
  - Machine Learning
categories: Learning Records
---

# Loss Functions

<!-- more -->

In machine learning, the "loss function" is a core concept used to quantify the difference between the predicted values of a model and the true values. The purpose of the loss function is to train the model by minimizing this difference, enabling it to make more accurate predictions.

Specifically, the loss function is usually defined as the accumulation or average of the differences between the model's predicted values and the true values. In regression problems, a commonly used loss function is the **Mean Squared Error (MSE)**, which calculates the average of the squares of the differences between the predicted value and the true value for each sample. In classification problems, a commonly used loss function is the **Cross-Entropy Loss**, which measures the difference between the probability distribution predicted by the model and the true distribution. By minimizing the loss function, we can find the parameter configuration that makes the model perform best on all samples.

## 1. Mean Squared Error (MSE)

The following is a detailed introduction to the Mean Squared Error (MSE) loss function, including its introduction, mathematical formula, working principle, Python code implementation, as well as its advantages and disadvantages.

### 1.1 Introduction

The Mean Squared Error loss function is one of the most commonly used loss functions in regression problems. Its purpose is to train the model by minimizing the squared difference between the predicted value and the true value, so that the model can predict results more accurately. MSE is a standard method for measuring the prediction performance of a model and is often used to evaluate the accuracy of regression models.

### 1.2 Mathematical Formula

The mathematical formula for the Mean Squared Error loss function is as follows:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:

- $n$ is the number of samples.
- $y_i$ is the true value of the $i$-th sample.
- $\hat{y}_i$ is the predicted value of the $i$-th sample.
- $\sum$ is the summation symbol, indicating that the sum is taken over all samples.
- $(y_i - \hat{y}_i)^2$ represents the square of the difference between the true value and the predicted value of the $i$-th sample.

### 1.3 Working Principle

The working principle of the MSE loss function is to evaluate the performance of the model by calculating the squared difference between the predicted value and the true value, summing these squared differences, and then averaging them. The training objective of the model is to minimize this average squared error value, so that the predicted values of the model are closer to the true values. By minimizing the MSE, the model can better fit the training data and improve the prediction accuracy.

### 1.4 Pure Python Code Implementation

In Python, the MSE loss function can be implemented using the NumPy library:

```python
import numpy as np
# True values
y_true = np.array([3, -0.5, 2, 7])
# Predicted values
y_pred = np.array([2.5, 0.0, 2, 8])
# Calculate MSE
mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)
```

### 1.5 Advantages and Disadvantages

**Advantages**:

- **Good Mathematical Properties**: MSE is a continuously differentiable convex function, ensuring that a global minimum can be found when using optimization algorithms such as gradient descent.
- **Large Penalty for Large Errors**: Due to the squared term, larger errors will have a greater impact on the loss function, which helps the model focus on data points that are particularly inaccurately predicted.

**Disadvantages**:

- **Sensitivity to Outliers**: Due to the squared error, outliers will have a disproportionate impact on the loss function, which may cause the model to be overly sensitive to outliers.

## 2. Mean Absolute Error (MAE)

### 2.1 Introduction

The Mean Absolute Error (MAE) is another commonly used loss function in regression problems. It evaluates the performance of the model by calculating the average of the absolute values of the differences between the predicted values and the true values. Compared with the Mean Squared Error (MSE), MAE is less sensitive to outliers, so it may be a better choice when there are outliers in the data.

### 2.2 Mathematical Formula

The mathematical formula for the Mean Absolute Error is as follows:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:

- $n$ is the number of samples.
- $y_i$ is the true value of the $i$-th sample.
- $\hat{y}_i$ is the predicted value of the $i$-th sample.
- $|\cdot|$ is the absolute value symbol.
- $\sum$ is the summation symbol, indicating that the sum is taken over all samples.

### 2.3 Working Principle

The working principle of the MAE loss function is to evaluate the performance of the model by calculating the absolute differences between the predicted values and the true values, summing these absolute differences, and then averaging them. The training objective of the model is to minimize this average absolute error value, so that the predicted values of the model are closer to the true values. By minimizing the MAE, the model can better fit the training data and improve the prediction accuracy.

### 2.4 Pure Python Code Implementation

In Python, the MAE loss function can be implemented using the NumPy library:

```python
import numpy as np
# True values
y_true = np.array([3, -0.5, 2, 7])
# Predicted values
y_pred = np.array([2.5, 0.0, 2, 8])
# Calculate MAE
mae = np.mean(np.abs(y_true - y_pred))
print("MAE:", mae)
```

### 2.5 Advantages and Disadvantages

**Advantages**:

- **Insensitivity to Outliers**: Due to the use of the absolute value, MAE is less sensitive to outliers, so it may be a better choice when there are outliers in the data.
- **Simple Calculation**: The calculation of MAE is relatively simple and only requires basic arithmetic operations.

**Disadvantages**:

- **Inability to Reflect the Magnitude of Errors**: MAE does not consider the absolute magnitude of the errors, so when the difference between the predicted value and the true value is large, MAE may not be able to accurately reflect the performance of the model.

## 3. Hinge Loss Function (Hinge Loss)

### 3.1 Introduction

The hinge loss function is a loss function used for classification problems, especially in Support Vector Machines (SVMs). Its purpose is to train the model by minimizing the loss of misclassified samples while keeping the loss of other samples at zero. The hinge loss function encourages the model to correctly classify the support vectors (i.e., the samples located near the decision boundary), and imposes a large loss on the misclassified samples.

### 3.2 Mathematical Formula

The mathematical formula for the hinge loss function is as follows:
For a binary classification problem, the formula is:

$$
L(y, f(x)) = \max(0, 1 - y f(x))
$$

Where:

- $y$ is the true label of the $i$-th sample (-1 or 1).
- $f(x)$ is the predicted score of the $i$-th sample.
- $\max(0, \cdot)$ represents taking the maximum value of the expression inside the parentheses, that is, only considering the non-negative part.

### 3.3 Working Principle

The working principle of the hinge loss function is to penalize the model by imposing a large loss on the misclassified samples, while the loss of the correctly classified samples is zero. In a binary classification problem, if the predicted score is greater than 1, the sample is considered a positive class; if the predicted score is less than -1, it is considered a negative class. If the predicted score is between -1 and 1, the sample is considered misclassified. The hinge loss function encourages the model to correctly classify the samples located near the decision boundary, that is, the support vectors, in this way.

### 3.4 Pure Python Code Implementation

In Python, the hinge loss function can be implemented using the NumPy library:

```python
import numpy as np
def hinge_loss(y_true, y_pred):
    """
    Calculate the value of the hinge loss function.
    :param y_true: True labels, a one-dimensional array or vector.
    :param y_pred: Predicted scores, a one-dimensional array or vector.
    :return: The value of the hinge loss function.
    """
    # Calculate the product of the predicted score and the true label
    margin = y_true * y_pred
    # Only consider the non-negative part
    loss = np.maximum(0, 1 - margin)
    # Calculate the average loss
    return np.mean(loss)
# Example data
y_true = np.array([1, -1, 1, -1])
y_pred = np.array([0.5, -0.5, 1.5, -0.5])
# Calculate the hinge loss
hinge_loss_value = hinge_loss(y_true, y_pred)
print("Hinge Loss:", hinge_loss_value)
```

### 3.5 Advantages and Disadvantages

**Advantages**:

- **Suitable for SVM**: The hinge loss function is the standard loss function in Support Vector Machines and is suitable for linearly separable and approximately linearly separable problems.
- **Insensitivity to Outliers**: Compared with the Mean Squared Error, the hinge loss is less sensitive to outliers.
- **Ability to Handle Nonlinear Problems**: By using the kernel trick, it can be extended to nonlinear problems.

**Disadvantages**:

- **Large Penalty for Misclassification**: The hinge loss imposes a large loss on misclassified samples, which may cause the model to become overly conservative during the training process.
- **Difficulty in Handling Multi-classification Problems**: The hinge loss function is mainly used for binary classification problems. When dealing with multi-classification problems, strategies such as One-vs-All or One-vs-One need to be used.
- **Parameter Sensitivity**: The regularization parameter C in the hinge loss function has a great impact on the performance of the model and needs to be adjusted carefully.

## 4. Exponential Loss Function (Exponential Loss)

### 4.1 Introduction

The exponential loss function, also known as a form of the log loss function, is a loss function commonly used in binary classification problems. It measures the difference between the predicted probability of the model and the true label. The exponential loss function encourages the model to make the predicted probability of positive samples close to 1 and the predicted probability of negative samples close to 0.

### 4.2 Mathematical Formula

The mathematical formula for the exponential loss function is as follows:
For a binary classification problem, the formula is:

$$
L(y, p) = -y \log(p) - (1 - y) \log(1 - p)
$$

Where:

- $y$ is the true label of the $i$-th sample (0 or 1).
- $p$ is the predicted probability of the positive class by the model.
- $\log$ is the natural logarithm.

### 4.3 Working Principle

The working principle of the exponential loss function is to penalize the predicted probability of positive samples if the predicted probability is less than the true label, and reward the predicted probability of negative samples if the predicted probability is greater than the true label. This penalty and reward mechanism enables the model to gradually adjust its parameters during the training process so that the predicted probability of positive samples approaches 1 and the predicted probability of negative samples approaches 0.

### 4.4 Pure Python Code Implementation

In Python, the exponential loss function can be implemented using the NumPy library:

```python
import numpy as np
def exponential_loss(y_true, y_pred):
    """
    Calculate the value of the exponential loss function.
    :param y_true: True labels, a one-dimensional array or vector.
    :param y_pred: Predicted probabilities, a one-dimensional array or vector.
    :return: The value of the exponential loss function.
    """
    # Calculate the exponential loss
    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    # Calculate the average loss
    return np.mean(loss)
# Example data
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.2, 0.8])
# Calculate the exponential loss
exponential_loss_value = exponential_loss(y_true, y_pred)
print("Exponential Loss:", exponential_loss_value)
```

### 4.5 Advantages and Disadvantages

**Advantages**:

- **Suitable for Binary Classification Problems**: The exponential loss function is suitable for binary classification problems and can effectively measure the predicted probabilities of positive and negative samples by the model.

**Disadvantages**:

- **Sensitivity to Predicted Probabilities**: The exponential loss function is very sensitive to small changes in the predicted probabilities, which may cause the model's adjustment of the predicted probabilities to be less smooth during the training process.
- **May Require Regularization**: In practical applications, the exponential loss function may need to add a regularization term to prevent overfitting.
Through the above introduction, you can have a more detailed understanding of the exponential loss function.

## 5. Huber Loss Function (Huber Loss)

### 5.1 Introduction

The Huber loss function is a commonly used loss function in regression problems. It combines the characteristics of the Mean Squared Error (MSE) and the absolute loss (MAE). When the error is small, the Huber loss function is close to the MSE, which ensures the continuous differentiability of the loss function; when the error is large, the Huber loss function becomes the absolute loss, which reduces the impact of large errors on the loss function. The Huber loss function is suitable for datasets containing outliers.

### 5.2 Mathematical Formula

The mathematical formula for the Huber loss function is as follows:

$$
L(a) = \begin{cases}
    \frac{1}{2}a^2 & \text{for } |a| \leq \delta \\
    \delta(|a| - \frac{1}{2}\delta) & \text{for } |a| > \delta
\end{cases}
$$

Where:

- $a$ is the difference between the predicted value and the true value.
- $\delta$ is the parameter of the Huber loss function, called "delta".

### 5.3 Working Principle

The working principle of the Huber loss function is to square the difference between the predicted value and the true value when the absolute value of the difference is less than or equal to delta; when the absolute value of the difference is greater than delta, use delta multiplied by the absolute value of the difference minus half of delta. This method makes the Huber loss function close to the Mean Squared Error when the difference between the predicted value and the true value is small, and close to the absolute loss when the difference is large.

### 5.4 Pure Python Code Implementation

In Python, the Huber loss function can be implemented using the NumPy library:

```python
import numpy as np
def huber_loss(y_true, y_pred, delta):
    """
    Calculate the value of the Huber loss function.
    :param y_true: True values, a one-dimensional array or vector.
    :param y_pred: Predicted values, a one-dimensional array or vector.
    :param delta: The parameter of the Huber loss function, that is, delta.
    :return: The value of the Huber loss function.
    """
    # Calculate the difference between the predicted value and the true value
    diff = y_true - y_pred
    # Calculate the absolute value of the difference
    diff_abs = np.abs(diff)
    # Determine whether the absolute value of the difference is greater than delta
    condition = diff_abs <= delta
    # When the absolute value of the difference is less than or equal to delta, use the Mean Squared Error
    mse = 0.5 * np.square(diff)
    # When the absolute value of the difference is greater than delta, use the absolute loss
    mae = delta * (diff_abs - 0.5 * delta)
    # Combine the two cases
    loss = np.where(condition, mse, mae)
    # Calculate the average loss
    return np.mean(loss)
# Example data
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
delta = 1.0
# Calculate the Huber loss
huber_loss_value = huber_loss(y_true, y_pred, delta)
print("Huber Loss:", huber_loss_value)
```

### 5.5 Advantages and Disadvantages

**Advantages**:

- **Insensitivity to Outliers**: The Huber loss function is insensitive to outliers and is suitable for datasets containing outliers.
- **Smooth Transition**: The Huber loss function smoothly transitions between the Mean Squared Error and the absolute loss, which can reduce the model's sensitivity to outliers.
- **Easy to Implement**: The implementation of the Huber loss function is relatively simple, and the delta parameter can be adjusted to adapt to different datasets.

**Disadvantages**:

- **Parameter Sensitivity**: The performance of the Huber loss function depends on the selection of the delta parameter. Improper selection may affect the performance of the model.
- **Computational Complexity**: Compared with the Mean Squared Error, the computational complexity of the Huber loss function is slightly higher because it is necessary to judge and calculate the difference of each sample