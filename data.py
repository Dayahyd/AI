import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# Step 1: Preparing the Features and Target Variable
# Load the wine dataset
wine_data = load_wine(as_frame=True)
df = wine_data.frame

# Assign features to x and target to y
x = df.drop('target', axis=1)
y = df['target']

# Step 2: Splitting the Data into Training and Testing Sets
# Split data with test_size=0.2 and random_state=42
x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 3: Creating and Training the Bagging Classifier Model
# Create a BaggingClassifier instance with random_state=1
bagging_model = BaggingClassifier(random_state=1)

# Fit the model to the training data
bagging_model.fit(x_train, y_train)

# Step 4: Making Predictions on the Training Set
# Predict on the training set
y_pred_train = bagging_model.predict(x_train)

# Step 5: Calculating Accuracy and Precision
# Calculate accuracy score on the training set
accuracy = accuracy_score(y_train, y_pred_train)

# Calculate precision score on the training set.
# 'weighted' is used for multiclass classification to account for class imbalance.
precision = precision_score(y_train, y_pred_train, average='weighted')

# Print the results
print("Accuracy on the training dataset:", accuracy)
print("Precision on the training dataset:", precision)
