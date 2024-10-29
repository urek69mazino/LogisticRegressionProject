# Import required libraries
from sklearn.datasets import load_breast_cancer 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np

# Load the breast cancer dataset and assign to variables
data = load_breast_cancer()
X = data.data     
y = data.target
feature_names = data.feature_names

# Convert to DataFrame for easier visualization and manipulation
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model and fit it to the training data
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

# Check model performance on the test data
y_pred = log_reg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to take user input for each feature
def get_user_input():
    print("Enter the values for the following features to predict if the tumor is malignant (1) or benign (0):")
    user_data = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        user_data.append(value)
    return np.array(user_data).reshape(1, -1)

# Take user input, predict the class, and output the result
user_data = get_user_input()
user_prediction = log_reg.predict(user_data)

# Display result based on the model's prediction
if user_prediction[0] == 1:
    print("\nThe tumor is predicted to be malignant.")
else:
    print("\nThe tumor is predicted to be benign.")
