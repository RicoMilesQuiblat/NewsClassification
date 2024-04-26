import streamlit as st
from keras.models import load_model
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("C:/Users/user/Downloads/News/news_classification")

def preprocess_input(input):
    input = input.lower()
    input = re.sub(r"[^\w\s]",input)
    input = nltk.word_tokenize(input)

    stop_words = set(stopwords.words("english"))
    input = [word for word in input if word not in stop_words]

    lemmatizer = WordNetLemmatizer;

    input = " ".join([lemmatizer.lemmatize(word) for word in input])

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(input)
    sequences = tokenizer.texts_to_sequences(input)
    features = pad_sequences(sequences)


def predict_category(input):
    processed_input = preprocess_input(input)
    prediction = model.predict(np.array([processed_input]))
    categories = ["WELLNESS", "ENTERTAINMENT", "POLITICS", "TRAVEL", "STYLE & BEAUTY"]
    return categories[np.argmax(prediction)]


def main():
    st.title('News Category Predictor')

    user_input = st.text_input("Enter a news headline:")

    if st.button('Predict'):
        prediction = predict_category(user_input)
        st.write(f'The predicted category is: {prediction}')


if __name__ == "__main__":
    main()
