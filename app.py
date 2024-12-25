# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import logging  # Add this

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

lemmatizer = WordNetLemmatizer()

# Define optimizer
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# chat initialization
model = load_model("chatbot_model.keras", compile=False)  # Load without optimizer state
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Load the intents and word/classes
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

dummy_input = np.zeros((1, len(words)))
dummy_output = np.zeros((1, len(classes)))

# Evaluating model with dummy data to check if it's loaded properly
model.evaluate(dummy_input, dummy_output, verbose=0)

# Save the model again in the current environment
model.save("chatbot_model_resaved.keras")

# Reload the model if needed (this is a workaround for any issues with model saving/loading)
model = load_model("chatbot_model_resaved.keras", compile=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

app = Flask(__name__)
#run_with_ngrok(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    logging.info(f"User Input: {msg}")  # Log the user input
    
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    
    return res

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    logging.debug(f"found in bag: {w}")  # Log debug details if needed
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    if not results:
        return [{"intent": "no_match", "probability": "0"}]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    result = "Sorry, I didn't get that."  # Fallback response if no match is found
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


if __name__ == "__main__":
    app.run(debug=True)