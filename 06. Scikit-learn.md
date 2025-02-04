🔍 Scikit-learn Cheatsheet 🧠
Scikit-learn is a powerful machine learning library in Python for data mining, data analysis, and machine learning algorithms.
________________________________________
🔹 1. Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
________________________________________
🔹 2. Load Data
# Load dataset from CSV or other formats
df = pd.read_csv("data.csv")
print(df.head())

# Features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]
________________________________________
🔹 3. Train-Test Split
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
________________________________________
🔹 4. Feature Scaling (Standardization)
# Standardize the features (important for algorithms like SVM, Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
________________________________________
🔹 5. Logistic Regression
# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
________________________________________
🔹 6. Support Vector Machine (SVM)
# SVM Model (Kernel type can be changed)
svm_model = SVC(kernel='linear')  # Use kernel='rbf' for non-linear
svm_model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
________________________________________
🔹 7. Decision Tree
# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
________________________________________
🔹 8. Random Forest
# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
________________________________________
🔹 9. K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
________________________________________
🔹 10. Cross-validation
from sklearn.model_selection import cross_val_score

# Cross-validation for a model
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean()}")
________________________________________
🔹 11. Grid Search for Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning using Grid Search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
________________________________________
🔹 12. PCA (Principal Component Analysis)
from sklearn.decomposition import PCA

# Apply PCA to reduce the dimensionality
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the 2D projection
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("PCA Plot")
plt.show()
________________________________________
🔹 13. Model Evaluation: Accuracy, Precision, Recall, F1-Score
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Precision, Recall, F1-Score
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
________________________________________
🔹 14. Confusion Matrix
from sklearn.metrics import confusion_matrix

# Confusion matrix for classification problems
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
________________________________________
🔹 15. ROC Curve and AUC
from sklearn.metrics import roc_curve, auc

# ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Receiver Operating Characteristic (ROC)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
________________________________________
🎯 Summary of Common Scikit-learn Models and Methods
Model/Method	Function
Logistic Regression	LogisticRegression()
Support Vector Machine (SVM)	SVC()
Decision Tree	DecisionTreeClassifier()
Random Forest	RandomForestClassifier()
K-Nearest Neighbors (KNN)	KNeighborsClassifier()
Train-Test Split	train_test_split()
Feature Scaling	StandardScaler()
Cross-validation	cross_val_score()
Hyperparameter Tuning	GridSearchCV()
PCA (Principal Component)	PCA()
Accuracy, Precision, Recall	accuracy_score(), precision_score(), recall_score()
Confusion Matrix	confusion_matrix()
ROC Curve	roc_curve(), auc()
________________________________________
🔥 Scikit-learn is a powerful and easy-to-use library for machine learning. Use these examples to get started with different algorithms and techniques! 🚀

