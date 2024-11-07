import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os

# Load dataset
data_path = 'Phishing_Legitimate_full.csv'  # Replace with your dataset path

try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The file {data_path} was not found. Please check the file path.")

# Check if 'label' column exists in dataset
if 'label' not in data.columns:
    raise KeyError("The dataset does not contain a 'label' column. Please check the dataset.")

# Features and target variable
X = data.drop('label', axis=1)  # Features (all columns except 'label')
y = data['label']  # Target variable ('label')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC()
}

# Store results
results = {}

# Loop through classifiers
for name, clf in classifiers.items():
    print(f"Training {name}...")
    
    # Define pipeline with scaling, PCA, and classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA(n_components=5)),  # Reduce dimensionality to 5 components
        ('classifier', clf)  # Classifier
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'classification_report': report
    }
    
    # Print results
    print(f"{name} - Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)
    print("-" * 50)

# Save the final model (Random Forest as an example)
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
pickle_file_name = 'best_model_phishing_detection.pkl'
model_path = os.path.join(model_dir, pickle_file_name)

best_model = classifiers['Random Forest'] 

# Save the best model
with open(model_path, 'wb') as model_file:
    pickle.dump(best_model, model_file)

print(f"Best model saved as '{pickle_file_name}' in the '{model_dir}' directory!")

# Optionally, print the comparison results of all classifiers
print("\nComparison of Classifier Performance:")
for name, result in results.items():
    print(f"{name} - Accuracy: {result['accuracy'] * 100:.2f}%")
    print(f"Classification Report:\n{result['classification_report']}")
    print("-" * 50)
