# 🧠 sem5_ML – Machine Learning Lab Assignments

This repository contains Python programs developed as part of the **23CSE301 Machine Learning Lab** coursework (Semester 5). Each file corresponds to a specific programming task, written with modular functions, inline documentation, and viva-friendly logic.

## 📂 Contents

| File Name     | Description                             |
|---------------|------------------------------------------|
| `lab1_q1.py`  | Count vowels and consonants in a string  |
| `lab1_q2.py`  | Matrix multiplication using nested loops |
| `lab1_q3.py`  | Find common elements between two lists   |
| `lab1_q4.py`  | Compute the transpose of a matrix        |
| `lab1_q5.py`  | Generate random numbers & calculate mean, median, mode |
| `lab2_A1.py`  | Segregate Purchase data into A and C matrices, find rank, dimension, and pseudo-inverse product prices  |
| `lab2_A2.py`  | Classify customers as RICH or POOR using Logistic Regression                                      |
| `lab2_A3.py`  | Analyze IRCTC stock price: mean, variance, April and Wed sample mean, probabilities, scatter plot                              |
| `lab2_A4.py`  | Data exploration on thyroid0387_UCI – attribute types, missing values, outliers, and stats    |
| `lab2_A5.py`  | Calculate Jaccard Coefficient (JC) and Simple Matching Coefficient (SMC) between two binary vectors                       |
| `lab2_A6.py`  | Cosine Similarity between first two complete numeric vectors from thyroid dataset                  |
| `lab2_A7.py`  |Compute JC, SMC, COS on first 20 observation vectors and display all in heatmaps        |
| `lab2_A8.py`  | Impute missing values using mean, median, or mode depending on type and outliers     |
| `lab2_A9.py`  | Verify and display that all missing values are now resolved (post A8)                           |
| `lab3_A1.py`  | Load feature vectors with class labels, drop filename column, and split dataset for classification tasks           |
| `lab3_A2.py`  | Train a k-Nearest Neighbors (kNN) classifier (k=3) and predict test set labels and individual sample classes        |
| `lab3_A3.py`  | Vary `k` from 1 to 11 in kNN to compare accuracy trends and visualize the effect of `k` with a line plot            |
| `lab3_A4.py`  | Compute confusion matrix and derive precision, recall, and F1-score for both training and test sets                |
| `lab3_A5.py`  | Generate and compare histogram and true PDF of normally distributed random values with KDE curve                  |
| `lab3_A6.py`  | Apply different distance metrics in kNN (e.g., Manhattan, Chebyshev) and observe classification behavior changes   |
| `lab3_A7.py`  | Plot AUROC curve for binary classification (Class 1 vs Class 2) using kNN and interpret AUC values                |
| `lab3_A8.py`  | Compare performance between manually developed kNN and package-based kNN classifier implementation                |
| `lab3_A9.py`   | Evaluate kNN classifier performance across training and test sets with learning outcome (underfit/regularfit/overfit) |
| `lab3_Q1.py`   | Calculate intraclass spread (mean and std) for each class based on feature vectors                       |
| `lab3_Q2.py`   | Compute interclass distance between class centroids using Euclidean and Minkowski metrics               |
| `lab3_Q3.py`   | Visualize histogram of feature value distribution per class along with mean and variance                 |
| `lab3_Q4.py`   | Plot Minkowski distance between two feature vectors for `r` varying from 1 to 10                          |
| `lab4_A1.py`  | Generate 2D synthetic training data with 20 points labeled as Class 0/1     |
| `lab4_A2.py`  | Train kNN classifier on synthetic data and classify 10,000 test points      |
| `lab4_A3.py`  | Visualize the decision boundary of kNN for various `k` values (1–6)         |
| `lab4_A4.py`  | Split project dataset and visualize training and test data distribution     |
| `lab4_A5.py`  | Plot decision boundaries on project data for various `k` values             |
| `lab4_A6.py`  | Evaluate classification with confusion matrices (train/test) and metrics    |
| `lab4_A7.py`  | Perform hyperparameter tuning using GridSearchCV / RandomizedSearchCV to find optimal `k` |
| `lab5_A1.py`  | Train linear regression model using `mfcc1` to predict confidence level                        |
| `lab5_A2.py`  | Evaluate A1 model using MSE, RMSE, R², and MAPE for train and test sets                        |
| `lab5_A3.py`  | Train and evaluate linear regression using all features (multivariate regression)              |
| `lab5_A4.py`  | Perform K-Means clustering with `k = 2` using all features, print cluster labels and centers    |
| `lab5_A5.py`  | Evaluate clustering (k=2) using Silhouette Score, Calinski-Harabasz Score, and DB Index        |
| `lab5_A6.py`  | Perform clustering for `k = 2 to 10`, plot Silhouette, CH, and DB scores separately             |
| `lab5_A7.py`  | Generate Elbow Plot using Inertia to determine the optimal number of clusters (`k`)            |
| `lab6_A1.py`           | Calculate entropy of target with equal-width binning; plot category counts. |
| `lab6_A2.py`           | Compute Gini index of target; plot category counts.                         |
| `lab6_A3.py`           | Calculate Information Gain for all features; select root feature; plot IG.  |
| `lab6_A4.py`           | Same as A3 but with parameterized binning (equal-width / equal-frequency).  |
| `lab6_A5.py`           | Build custom recursive Decision Tree; display structure in pretty text form.|
| `lab6_A6.py`           | Train sklearn DecisionTreeClassifier and visualize tree with plot_tree().   |
| `lab6_A7.py`           | Use two features to train DT and plot 2D decision boundary with regions.    |
| `lab7_A2.py`  | Implements A2: Hyperparameter tuning with RandomizedSearchCV (KNN, SVM, DecisionTree, RandomForest). Saves per-model reports, CV plots, a summary CSV, and a comparison bar chart. |
| `lab7_A3.py`  | Implements A3: Classification comparison (SVM, DecisionTree, RandomForest, AdaBoost, Naïve Bayes, MLP; XGBoost/CatBoost if installed). Produces a Train vs Test metrics table and Test-F1 bar plot. |
| `lab7_A4.py`  | Placeholder for A4 (Regression). Not required for your dataset since it is classification-based. |
| `lab7_Q1.py`  | Implements O1: SHAP explainability. Generates SHAP bar/beeswarm plots for RandomForest and Logistic Regression and a feature-importance comparison CSV. |
| `lab7_Q2.py`  | Implements O2: LIME local explanations. Generates a PNG explanation of a chosen test instance prediction from RandomForest. |




## ✅ Features

- Written in pure Python (no external libraries)
- Fully modular and plagiarism-safe
- Inline comments for better understanding
- Git-tracked for version history

## 📚 Course Info

- **Subject Code:** 23CSE301  
- **Semester:** V  
- **Instructor:** Dr. Peeta Basa Pati  
- **College:** Amrita Vishwa Vidyapeetham, Bangalore

## 🚀 How to Run

Use any Python 3 interpreter or IDE. Each file takes input from the user and displays results to the terminal.

```bash
python lab1_q2.py
