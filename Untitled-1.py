from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

# Output results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Visualization
plt.figure(figsize=(15, 10))

# 1. Decision Tree Visualization
plt.subplot(2, 2, 1)
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, 
          filled=True, rounded=True, fontsize=8)
plt.title('Decision Tree Structure')

# 2. Confusion Matrix
plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 3. Feature Importance
plt.subplot(2, 2, 3)
importances = clf.feature_importances_
sns.barplot(x=iris.feature_names, y=importances)
plt.title('Feature Importance')
plt.xticks(rotation=45)

# 4. Scatter Plot of two most important features
plt.subplot(2, 2, 4)
top_features = np.argsort(importances)[-2:]
sns.scatterplot(x=X_test[:, top_features[0]], 
                y=X_test[:, top_features[1]], 
                hue=[iris.target_names[i] for i in y_test])
plt.xlabel(iris.feature_names[top_features[0]])
plt.ylabel(iris.feature_names[top_features[1]])
plt.title('Scatter Plot of Top 2 Important Features')

plt.tight_layout()
plt.show()
