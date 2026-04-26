import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("all_watch_data_4hz.csv")

# Remove rows with missing values
df = df.dropna()

# Create binary pain label:
# 0 = no pain
# 1 = pain
df["pain_label"] = df["pain_scale"].apply(lambda x: 0 if x == 0 else 1)

# Features we are using
# Excluding x, y, z because those are accelerometer / movement data
X = df[["bvp", "eda", "temperature"]]

# Target label
y = df["pain_label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build random forest model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print results
print("Random Forest Pain vs No Pain Model")
print("-----------------------------------")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))
