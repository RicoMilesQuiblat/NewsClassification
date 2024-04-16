import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import  train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
pd.options.display.max_rows = 10
filename = "C:/Users/user/Downloads/News/News.json"
df = pd.read_json(filename, lines=True)


df = df.dropna()
df = df.drop(columns=["link"])
df = df.drop(columns=["date"])
df = df.drop(columns=["authors"])
categories = ["SPORTS", "POLITICS", "BUSINESS", "CRIME", "SCIENCE", "TECH"]
df = df[df["category"].isin(categories)]

num_rows = len(df)
half = num_rows // 2
df = df.iloc[:half // 4]

df["headline"] = df["headline"].str.lower()
df["category"] = df["category"].str.lower()
df["short_description"] = df["short_description"].str.lower()

df['headline'] = df['headline'].apply(lambda x: re.sub(r"[^\w\s]", "", x))
df['short_description'] = df['short_description'].apply(lambda x: re.sub(r"[^\w\s]", "", x))

df["headline"] = df["headline"].apply(nltk.word_tokenize)
df["short_description"] = df["short_description"].apply(nltk.word_tokenize)

stop_words = set(stopwords.words("english"))

df["headline"] = df["headline"].apply(lambda x: [word for word in x if word not in stop_words])
df["short_description"] = df["short_description"].apply(lambda x: [word for word in x if word not in stop_words])

lemmatizer = WordNetLemmatizer()

df["headline"] = df["headline"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x]))
df["short_description"] = df["short_description"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x]))

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df["headline"] + " " + df["short_description"])
sequences = tokenizer.texts_to_sequences(df["headline"] + " " + df["short_description"])

features = pad_sequences(sequences)

enc = OneHotEncoder(handle_unknown="ignore")
category = enc.fit_transform(df[["category"]]).toarray()

x_train, x_test, y_train, y_test = train_test_split(features, category, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1

output_size = category.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(Bidirectional(LSTM(units=64)))
model.add(Dropout(rate=0.5))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(output_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

history = model.fit(x_train,y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=False)

model.evaluate(x_test, y_test)


