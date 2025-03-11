from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from main import clean_data

# 1) Preparing data for models
X = clean_data['statement'] #Predictor variable
y = clean_data['status_encoded'] #Target variable

# 2) Transform text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# 3) Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Create and train the Decision Tree model
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(X_train, y_train)

# 5) Make predictions on the testing set
y_pred_tree = model_tree.predict(X_test)

# 6) Calculate the evaluation metrics
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)

# 7) Print the evaluation results
print("\nDecision Tree Model evaluation results:")
print(f"Accuracy: {accuracy_tree:.2%}")
print(f"Precision: {precision_tree:.2%}")
print(f"Recall: {recall_tree:.2%}")

# 8) ROC for Decision Tree
from sklearn.metrics import roc_curve, auc

## Predicted probabilities
y_pred_probs_tree_d = model_tree.predict_proba(X_test)[:,0] #Probability of class 0 (Depression)
y_pred_probs_tree_n = model_tree.predict_proba(X_test)[:,1] #Probability of class 1 (Normal)

## Calculate ROC
### For depression (class 0)
fpr_tree_d, tpr_tree_d, thresholds_tree_d = roc_curve(y_test, y_pred_probs_tree_d)
roc_auc_tree_d = auc(fpr_tree_d, tpr_tree_d)

### For normal (class 1)
fpr_tree_n, tpr_tree_n, thresholds_tree_n = roc_curve(y_test, y_pred_probs_tree_n)
roc_auc_tree_n = auc(fpr_tree_n, tpr_tree_n)

## Plot ROC Curve for Decision Tree
import matplotlib.pyplot as plt

### Probability of class 0 (Depression)
plt.figure(figsize=(8, 6))
plt.plot(fpr_tree_d, tpr_tree_d, color='darkorange', lw=2, label=f"ROC curve (AUC={roc_auc_tree_d:.2%})")
plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Decision Tree (Depression)")
plt.legend(loc="lower right")
plt.show()

### Probability of class 1 (Normal)
plt.figure(figsize=(8, 6))
plt.plot(fpr_tree_n, tpr_tree_n, color='darkorange', lw=2, label=f"ROC curve (AUC={roc_auc_tree_n:.2%})")
plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Decision Tree (Normal)")
plt.legend(loc="lower right")
plt.show()