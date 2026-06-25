import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("ml/features/features.csv")

# Features
X = df.drop("speaker", axis=1)

# Labels
y = df["speaker"]

# Convert names to numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nSpeaker Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{i} -> {label}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

import joblib

joblib.dump(model, "ml/models/speaker_model.pkl")
joblib.dump(encoder, "ml/models/label_encoder.pkl")

print("\nModel Saved Successfully")