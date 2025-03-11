# Predicts the sentiment label for a given text
## Using Neural Network Model
### 0 for 'Depression', 1 for 'Normal'

from decision_tree import vectorizer
from neural_network import model_nn

# 1) Create a function using Neural Network Model
def predict_sentiment_nn(text):
    text_vectorized = vectorizer.transform([text])
    prediction_nn = model_nn.predict(text_vectorized)[0]
    if prediction_nn == 0:
        return "Depression"
    else:
        return "Normal"

# 2) Print the test results for Neural Network
print("\nNeural Network Model test results:")
print("Statement 1: ", predict_sentiment_nn("I can't stop worrying about everything"))
print("Statement 2: ", predict_sentiment_nn("I've been working hard, and seeing the results makes me feel incredibly happy and fulfilled"))
print("Statement 3: ", predict_sentiment_nn("Even the smallest things feel like too much right now"))
print("Statement 4: ", predict_sentiment_nn("I can’t stop smiling"))
print("Statement 5: ", predict_sentiment_nn("Today has been amazing!"))