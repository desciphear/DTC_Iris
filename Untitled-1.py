from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
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

# Create subplots
fig = sp.make_subplots(rows=2, cols=2,
                       subplot_titles=('Decision Tree Structure', 'Confusion Matrix',
                                     'Feature Importance', 'Top 2 Important Features'))

# 1. Decision Tree Visualization (Note: plotly doesn't have direct tree visualization)
# You might want to use graphviz for this separately

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
heatmap = go.Heatmap(z=cm,
                     x=iris.target_names,
                     y=iris.target_names,
                     text=cm,
                     texttemplate="%{text}",
                     textfont={"size": 16},
                     colorscale='Blues')
fig.add_trace(heatmap, row=1, col=2)

# 3. Feature Importance
importances = clf.feature_importances_
bar = go.Bar(x=iris.feature_names,
             y=importances)
fig.add_trace(bar, row=2, col=1)

# 4. Scatter Plot of two most important features
top_features = np.argsort(importances)[-2:]
scatter = go.Scatter(x=X_test[:, top_features[0]],
                    y=X_test[:, top_features[1]],
                    mode='markers',
                    marker=dict(color=y_test),
                    text=[iris.target_names[i] for i in y_test])
fig.add_trace(scatter, row=2, col=2)

# Update layout
fig.update_layout(height=800, width=1200, showlegend=False,
                 title_text="Iris Decision Tree Analysis")

fig.update_xaxes(title_text=iris.feature_names[top_features[0]], row=2, col=2)
fig.update_yaxes(title_text=iris.feature_names[top_features[1]], row=2, col=2)

# Show the plot
fig.show()
