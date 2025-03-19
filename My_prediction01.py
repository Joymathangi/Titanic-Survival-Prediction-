import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('tested.csv')

# Preprocessing
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = data[features].copy()  # Create a copy to avoid view issues
y = data['Survived']

# Normalize using .loc to avoid warning
scaler = StandardScaler()
X.loc[:, ['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Feature importance plot
plt.barh(features, model.feature_importances_)
plt.xlabel('Feature Importance')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.show()