import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib

# Load the data
df = pd.read_csv('EEG Eye State.csv')

# Check and clean target column if necessary
if df['eyeDetection'].dtype != 'int64':
    df['eyeDetection'] = df['eyeDetection'].astype(int)

# Features and Target
X = df.drop('eyeDetection', axis=1)
y = df['eyeDetection']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'best_eye_state_model.pkl')
print("Best model saved as 'best_eye_state_model.pkl'.")

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Feature Importance Plot
importances = best_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (EEG Sensors)")
plt.bar(range(X.shape[1]), importances[indices], color="skyblue", align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importance_plot.png')
plt.show()
