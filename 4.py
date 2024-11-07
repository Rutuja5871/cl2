import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("datasets/polymerase_cluster.csv")

# Extract feature columns (assuming they are named as '0', '1', ..., '29')
X = df[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
        '21', '22', '23', '24', '25', '26', '27', '28', '29']]

# Assuming 'G' columns represent classes, combine them into a single target variable
# Convert multi-class one-hot encoding into a single categorical target
df['target'] = df[['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']].idxmax(axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rfc.predict(X_test)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cr = classification_report(y_test, y_pred)
print("Classification report:\n", cr)

# Define a new input array (should match the number of features, 30 in this case)
# Example: Custom input for a single prediction
custom_input = [[0.5, 1.2, -0.3, 2.1, 3.2, 0.1, -1.0, 0.4, 1.8, -0.5, 
                 0.2, 2.3, -1.2, 1.1, 0.5, 1.7, 2.6, -0.7, 1.0, -0.3,
                 0.6, -1.4, 1.1, -0.8, 0.5, 1.9, 0.2, -1.6, 2.3, 0.9]]

# Predict with custom input
custom_pred = rfc.predict(custom_input)
print("Prediction for custom input:", custom_pred)
