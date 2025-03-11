from sklearn.neural_network import MLPClassifier
from decision_tree import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1) Create and train the Neural Network model
model_nn = MLPClassifier(
    hidden_layer_sizes=(50,), max_iter=100, random_state=42)
model_nn.fit(X_train, y_train)

# 2) Make predictions on the testing set
y_pred_nn = model_nn.predict(X_test)

# 3) Calculate the evaluation metrics
accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)

# 4) Print the evaluation results
print("\nNeural Network Model evaluation results:")
print(f"Accuracy: {accuracy_nn:.2%}")
print(f"Precision: {precision_nn:.2%}")
print(f"Recall: {recall_nn:.2%}")

# 5) ROC for Neural Network Model
from sklearn.metrics import roc_curve, auc

## Predicted probabilities
y_pred_probs_nn_d = model_nn.predict_proba(X_test)[:,0] #Probability of class 0 (Depression)
y_pred_probs_nn_n = model_nn.predict_proba(X_test)[:,1] #Probability of class 1 (Normal)

## Calculate ROC
### For depression (class 0)
fpr_nn_d, tpr_nn_d, thresholds_nn_d = roc_curve(y_test, y_pred_probs_nn_d)
roc_auc_nn_d = auc(fpr_nn_d, tpr_nn_d)

### For normal (class 1)
fpr_nn_n, tpr_nn_n, thresholds_nn_n = roc_curve(y_test, y_pred_probs_nn_n)
roc_auc_nn_n = auc(fpr_nn_n, tpr_nn_n)

## Plot ROC Curve for Decision Tree
import matplotlib.pyplot as plt

### Probability of class 0 (Depression)
plt.figure(figsize=(8, 6))
plt.plot(fpr_nn_d, tpr_nn_d, color='darkorange', lw=2, label=f"ROC curve (AUC={roc_auc_nn_d:.2%})")
plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Neural Network (Depression)")
plt.legend(loc="lower right")
plt.show()

### Probability of class 1 (Normal)
plt.figure(figsize=(8, 6))
plt.plot(fpr_nn_n, tpr_nn_n, color='darkorange', lw=2, label=f"ROC curve (AUC={roc_auc_nn_n:.2%})")
plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Neural Network (Normal)")
plt.legend(loc="lower right")
plt.show()

