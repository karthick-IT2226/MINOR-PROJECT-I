import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

data = pd.read_csv('Phishing_Legitimate_full.csv')

print("First few rows of the dataset:")
print(data.head())
print("\nColumn names in the dataset:")
print(data.columns)

if 'label' not in data.columns:
    raise KeyError("The dataset does not contain a 'label' column. Please check the dataset.")

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

pickle_file_name = 'phishing_detection_model_v1.pkl'
model_path = os.path.join(model_dir, pickle_file_name)

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model saved successfully as '{pickle_file_name}' in the '{model_dir}' directory!")
