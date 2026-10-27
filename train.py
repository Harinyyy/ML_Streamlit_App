
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris flower dataset
iris = load_iris()

# Separate input features and target labels
X = iris.data
y = iris.target

# Create the machine learning model
model = RandomForestClassifier()

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "iris_model.pkl")

print("Model trained and saved successfully!")
