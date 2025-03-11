# Predicts the sentiment label for a given text
## Using Decision Tree Model
### 0 for 'Depression', 1 for 'Normal'

from decision_tree import vectorizer, model_tree

# 1) Create a function using Decision Tree Model
def predict_sentiment_tree(text):
    text_vectorized = vectorizer.transform([text])
    prediction_tree = model_tree.predict(text_vectorized)[0]
    if prediction_tree == 0:
        return "Depression"
    else:
        return "Normal"

# 2) Print the test results for Decision Tree
print("\nDecision Tree Model test results:")
print("Statement 1: ", predict_sentiment_tree("I can't stop worrying about everything"))
print("Statement 2: ", predict_sentiment_tree("I've been working hard, and seeing the results makes me feel incredibly happy and fulfilled"))
print("Statement 3: ", predict_sentiment_tree("Even the smallest things feel like too much right now"))
print("Statement 4: ", predict_sentiment_tree("I can’t stop smiling"))
print("Statement 5: ", predict_sentiment_tree("Today has been amazing!"))