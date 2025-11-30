1. Import necessary libraries python
import pandas as pd
import Numphy as num
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow_datasets as tfds
Use code with caution.

pandas: Used for data manipulation and analysis, primarily for creating and working with DataFrames.
sklearn: The scikit-learn library, a popular tool for machine learning in Python.
tensorflow_datasets: Provides a collection of ready-to-use datasets, including Titanic, which simplifies the data loading process. 
2. Load and convert the dataset
python
dataset, info = tfds.load('titanic', split='train', with_info=True)
df = tfds.as_dataframe(dataset, info)
Use code with caution.

tfds.load('titanic', ...): Downloads and loads the Titanic dataset from TensorFlow Datasets.
split='train': Specifies that only the training split of the dataset should be loaded.
with_info=True: Returns a tuple containing the dataset and its metadata.
tfds.as_dataframe(dataset, info): Converts the TensorFlow dataset into a pandas DataFrame, making it easier to work with using familiar pandas functions. 
3. Prepare features and target variable
python
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
y = df['survived']
X['age'] = X['age'].fillna(X['age'].median())
X = pd.get_dummies(X, columns=['sex'], drop_first=True)
Use code with caution.

X = ...: Creates the feature set X by selecting a subset of columns from the DataFrame. These columns are the features the model will use to make predictions.
y = ...: Creates the target variable y, which is the survived column. This is what the model is trying to predict (1 for survived, 0 for not survived).
X['age'] = X['age'].fillna(...): Fills in missing values (NaN) in the 'age' column with the median age. This is a common strategy for handling missing data.
X = pd.get_dummies(...): Converts the categorical 'sex' column into numerical format using one-hot encoding.
drop_first=True: Creates sex_male as a new column, where 1 indicates male and 0 indicates female. This avoids multicollinearity. 
4. Split the data
python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Use code with caution.

train_test_split(...): Divides the data into training and testing sets.
X_train and y_train: Used to train the model (80% of the data).
X_test and y_test: Used to evaluate the model (20% of the data).
test_size=0.2: Allocates 20% of the data to the test set.
random_state=42: Ensures the split is consistent and reproducible each time the code is run. 
5. Train the model
python
model = LogisticRegression()
model.fit(X_train, y_train)
Use code with caution.

model = LogisticRegression(): Creates an instance of the LogisticRegression model, which is suitable for binary classification problems like this one.
model.fit(X_train, y_train): Trains the model using the training data. The model learns the relationship between the features (X_train) and the target variable (y_train). 
6. Make predictions and evaluate accuracy
python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Use code with caution.

y_pred = model.predict(X_test): Uses the trained model to make predictions on the unseen test data (X_test).
accuracy = accuracy_score(y_test, y_pred): Compares the model's predictions (y_pred) with the actual target values (y_test) and calculates the accuracy score. This score represents the proportion of correctly predicted instances. 


