import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset
data = pd.read_csv('datasets/emails.csv', sep=',')  # Adjust the path to your CSV file
data.columns = ['text', 'spam']  # Renaming columns for easier access

# Display the first few rows to understand the dataset
print(data.head())

# Step 2: Preprocessing
# Convert 'spam' column to binary values
data['label'] = data['spam'].apply(lambda x: 1 if x == 1 else 0)  # Spam=1, Not Spam=0

# Step 3: Split the data
X = data['text']  # Features
y = data['label']  # Target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a pipeline with TfidfVectorizer and Random Forest Classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = pipeline.predict(X_test)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Testing with unknown input data
# Example unknown input data
unknown_emails = [
    "Congratulations! You've won a $1000 cash prize. Click here to claim your money!",
    "Dear customer, your subscription is about to expire. Please renew now."
]

# Predicting the class of the unknown input data
predictions = pipeline.predict(unknown_emails)
predicted_labels = ["spam" if label == 1 else "not spam" for label in predictions]

# Displaying predictions for unknown emails
for email, label in zip(unknown_emails, predicted_labels):
    print(f"Email: {email}\nPredicted label: {label}\n")
