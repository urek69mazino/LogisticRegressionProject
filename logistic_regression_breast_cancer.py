# Import the breast cancer dataset from sklearn datasets module
from sklearn.datasets import load_breast_cancer 

# Import pandas for data manipulation and visualization
import pandas as pd

# Load the breast cancer dataset and store it in the variable `data`
data = load_breast_cancer()

# Assign the features (input variables) of the dataset to `X`
X = data.data

# Assign the target (output variable) of the dataset to `y`
y = data.target

# Retrieve the names of each feature (column) in the dataset for labeling purposes
feature_names = data.feature_names

# Convert the features and target into a DataFrame for easier visualization
df = pd.DataFrame(X, columns=feature_names)

# Add the target variable to the DataFrame, creating a new column labeled 'target'
df['target'] = y

# Display the first few rows of the DataFrame to verify its structure
df.head()

# Import the train_test_split function to split the data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets; 80% for training and 20% for testing
# `random_state=42` ensures the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import the LogisticRegression model from sklearn
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model with a maximum of 10,000 iterations
log_reg = LogisticRegression(max_iter=10000)

# Train (fit) the Logistic Regression model on the training data
log_reg.fit(X_train, y_train)

# Import evaluation metrics for the model, including accuracy, precision, recall, confusion matrix, and classification report
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Use the trained model to make predictions on the test data
y_pred = log_reg.predict(X_test)

# Calculate the accuracy of the model by comparing predictions to the actual test labels
accuracy = accuracy_score(y_test, y_pred)

# Calculate the precision, which measures how many positive predictions are correct
precision = precision_score(y_test, y_pred)

# Calculate the recall, which measures how many actual positives are correctly predicted
recall = recall_score(y_test, y_pred)

# Create a confusion matrix to show the counts of true positive, true negative, false positive, and false negative predictions
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate a classification report summarizing precision, recall, and F1-score for each class
report = classification_report(y_test, y_pred)

# Print the accuracy score of the model
print(f"Accuracy: {accuracy}")

# Print the precision score of the model
print(f"Precision: {precision}")

# Print the recall score of the model
print(f"Recall: {recall}")

# Print the confusion matrix to observe the breakdown of predictions
print("Confusion Matrix:\n", conf_matrix)

# Print the classification report for a more detailed evaluation of model performance
print("Classification Report:\n", report)

# Import the ROC curve and AUC score functions
from sklearn.metrics import roc_curve, roc_auc_score

# Calculate the probabilities of each test sample belonging to the positive class
# `predict_proba` returns probabilities for both classes, and we take the probability for class 1 (malignant)
y_probs = log_reg.predict_proba(X_test)[:, 1]

# Compute the false positive rate, true positive rate, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate the AUC score, a single value summary of the ROC curve’s performance
roc_auc = roc_auc_score(y_test, y_probs)

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Plot the ROC curve with the false positive rate on the x-axis and the true positive rate on the y-axis
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')

# Plot a diagonal line representing a model with no discrimination power (random guessing)
plt.plot([0, 1], [0, 1], 'k--')

# Label the x-axis as "False Positive Rate"
plt.xlabel('False Positive Rate')

# Label the y-axis as "True Positive Rate"
plt.ylabel('True Positive Rate')

# Title the plot as "Receiver Operating Characteristic (ROC) Curve"
plt.title('Receiver Operating Characteristic (ROC) Curve')

# Add a legend to the plot
plt.legend()

# Display a grid for easier visualization of the ROC curve
plt.grid()

# Show the plot
plt.show()
