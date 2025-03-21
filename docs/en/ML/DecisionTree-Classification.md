---
title: Decision Tree Classification Algorithm Principles
description: Principles of the decision tree classification algorithm, including Example 1, Example 2, and characteristics of the decision tree algorithm.
date: 2021-06-12
tags:
  - Algorithms
  - Machine Learning
categories: Study Notes
cover: /post/DecisionTree-Classification/1-决策树.png
hiddenCover: true
hidden: false
readingTime: true
comment: false
author: Adream
---

# Decision Tree

<!-- more -->

## 1. Overview of Decision Trees

Decision trees belong to supervised machine learning, originating early in AI development. They mimic human decision-making processes and are intuitive and interpretable. While early AI models widely used decision trees, modern applications often leverage ensemble methods based on decision trees. Understanding decision trees thoroughly is crucial for learning ensemble methods.

### 1.1 Example 1

Consider the following loan data:

| ID  | Owns Property (Yes/No) | Marital Status [Single, Married, Divorced] | Annual Income (k) | Defaults on Debt (Yes/No) |
| --- | ---------------------- | ------------------------------------------ | ----------------- | ------------------------- |
| 1   | Yes                    | Single                                     | 125               | No                        |
| 2   | No                     | Married                                    | 100               | No                        |
| 3   | No                     | Single                                     | 70                | No                        |
| 4   | Yes                    | Married                                    | 120               | No                        |
| 5   | No                     | Divorced                                   | 95                | Yes                       |
| 6   | No                     | Married                                    | 60                | No                        |
| 7   | Yes                    | Divorced                                   | 220               | No                        |
| 8   | No                     | Single                                     | 85                | Yes                       |
| 9   | No                     | Married                                    | 75                | No                        |
| 10  | No                     | Single                                     | 90                | Yes                       |

Based on this historical data, we build a decision tree to predict whether a user will default:

![](/post/DecisionTree-Classification/1-决策树.png)

For a new user: No property, Single, 55K income, the decision tree predicts default (blue dashed path). The tree also reveals that property ownership significantly impacts debt repayment, guiding loan decisions.

### 1.2 Example 2

A mother arranges a blind date for her daughter. Key decision factors include age, appearance, income, and occupation. The decision tree might look like:

![](/post/DecisionTree-Classification/2-相亲.png)

**Note**:  
- **Features**: Green nodes (age, appearance, income, civil servant status).  
- **Target**: Orange nodes (decision).  
- Different individuals may construct different trees based on priorities, but algorithms follow consistent criteria.

### 1.3 Characteristics of Decision Tree Algorithms
- Handle nonlinear problems.
- Strong interpretability (no coefficients like in linear models).
- Simple structure with efficient prediction via `if-else` logic.

## 2. Using `DecisionTreeClassifier`

### 2.1 Problem: Authenticating Social Media Accounts

![](/post/DecisionTree-Classification/3-账号真伪.png)

Predict account authenticity based on:  
- **Log Density** (s: small, m: medium, l: large)  
- **Friend Density** (s, m, l)  
- **Uses Real Avatar** (Y/N)  

### 2.2 Building and Visualizing the Tree

**Data Preparation**  
```python
import numpy as np
import pandas as pd

y = np.array(list('NYYYYYNYYN'))
X = pd.DataFrame({
    'Log Density': list('sslmlmmlms'),
    'Friend Density': list('slmmmlsmss'),
    'Real Avatar': list('NYYYYNYYYY'),
    'Authentic User': y
})

# Convert categorical features to numerical
X['Log Density'] = X['Log Density'].map({'s':0, 'm':1, 'l':2})
X['Friend Density'] = X['Friend Density'].map({'s':0, 'm':1, 'l':2})
X['Real Avatar'] = X['Real Avatar'].map({'N':0, 'Y':1})
```

**Model Training & Visualization**  
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X.drop('Authentic User', axis=1), y)

plt.figure(figsize=(12,16))
plot_tree(model, filled=True, feature_names=X.columns[:-1], class_names=['Fake','Real'])
plt.savefig('./account_tree.png')
```

![](/post/DecisionTree-Classification/4-account.jpg)

**Graphviz Visualization**  
```python
import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(model, out_file=None,
                           feature_names=X.columns[:-1],
                           class_names=['Fake','Real'],
                           filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render('account_decision_tree')
```

### 2.3 Information Entropy

Entropy quantifies uncertainty in data:  
$$\text{H}(X) = -\sum_{i=1}^n p(x_i) \log_2 p(x_i)$$

### 2.4 Information Gain

Information gain measures uncertainty reduction after splitting by a feature:  
$$\text{IG}(Y, X) = \text{H}(Y) - \text{H}(Y|X)$$

### 2.5 Manual Calculation for Splitting

**Step 1: Compute Base Entropy**  
```python
s = X['Authentic User']
p = s.value_counts()/s.size
base_entropy = -(p * np.log2(p)).sum()  # ≈ 0.97
```

**Step 2: Evaluate Splits for Each Feature**  
For `Friend Density`:  
```python
splits = [0.5]  # Midpoint between 0 and 1
for split in splits:
    cond = X['Friend Density'] <= split
    p_cond = cond.value_counts(normalize=True)
    entropy = 0
    for idx in p_cond.index:
        subset = X[cond == idx]['Authentic User']
        p_subset = subset.value_counts(normalize=True)
        entropy += p_cond[idx] * -(p_subset * np.log2(p_subset)).sum()
    print(f"Split at {split}: Entropy={entropy:.3f}")
```

**Optimal Split**: Choose the feature and split point with the lowest entropy.

## 3. Splitting Criteria

### 3.1 Information Gain (ID3)
Maximize information gain to select splits. Prone to bias toward high-cardinality features.

### 3.2 Gini Index (CART)
Gini impurity measures inequality:  
$$\text{Gini} = \sum_{i=1}^n p_i(1 - p_i)$$  
Lower Gini indicates purer nodes.

### 3.3 Gain Ratio (C4.5)
Adjusts information gain by intrinsic value of the feature:  
$$\text{GainRatio} = \frac{\text{IG}(Y, X)}{\text{IV}(X)}$$  
where $$\text{IV}(X) = -\sum \frac{|X_v|}{|X|} \log_2 \frac{|X_v|}{|X|}$$.

### 3.4 MSE (Regression Trees)
Used for regression by minimizing mean squared error.

## 4. Iris Classification with Decision Trees

### 4.1 Model Training
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

### 4.2 Visualizing the Tree
![](/post/DecisionTree-Classification/12-iris.png)

### 4.3 Pruning
```python
model = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.1, max_depth=3)
model.fit(X_train, y_train)
```

### 4.4 Hyperparameter Tuning
```python
import matplotlib.pyplot as plt

depths = range(1, 16)
errors = []
for d in depths:
    model = DecisionTreeClassifier(max_depth=d)
    model.fit(X_train, y_train)
    errors.append(1 - model.score(X_test, y_test))

plt.plot(depths, errors, 'ro-')
plt.xlabel('Tree Depth')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. Tree Depth')
plt.grid()
plt.show()
```

![](/post/DecisionTree-Classification/14-筛选超参数.png)

### 4.5 Feature Importance
```python
print("Feature Importances:", model.feature_importances_)
```