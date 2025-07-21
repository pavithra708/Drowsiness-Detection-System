import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
import joblib

# Step 1: Load dataset
df = pd.read_csv('dataset.csv')
print("📊 Dataset loaded:")
print(df.head())

# Step 2: Clean labels
df['label'] = df['label'].str.strip().str.lower()
print("✅ Cleaned labels:", df['label'].unique())

# Step 3: Split into X and y
X = df[['EAR', 'MAR', 'blink_rate']]
y = df['label']

# Step 4: Encode
y_encoded = y.map({'alert': 0, 'drowsy': 1})
print("✅ Encoded labels:", y_encoded.unique())

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Step 6: Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("✅ Model trained!")
# ✅ Step 6: Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ✅ Step 7: Save trained model to file
joblib.dump(clf, 'model.pkl')
print("📦 Model saved to model.pkl ✅")
