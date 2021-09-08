import numpy as np
import pandas as pd
import csv

from datetime import datetime
from datetime import date

import streamlit as st

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import load_model

train_data = pd.read_csv("tweet_emotions.csv")
train_data.head()

training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "content"]
    training_sentences.append(sentence)


model = load_model("Tweets_Text_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)


################################################################################################################
date_time = date.today().strftime("%A %d %B %Y")


def save(text, emotion):

    day_entry = {"date": date_time, "text": text, "emotion": emotion}

    with open("data_entry.csv", "a") as f:
        # for key in day_entry.keys():
        f.write("%s,%s,%s\n" % (date_time,text, emotion))


# Title
st.title("Digital Journal")
# Day and Date
st.write(date_time)

# New Entry
text_input = str(st.text_input("How was your day?"))

if text_input != "":
    sentence = []
    sentence.append(text_input)
    print(sentence)

    sequences = tokenizer.texts_to_sequences(sentence)
    print("sequence", sequences)

    padded = pad_sequences(
        sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )
    testing_padded = np.array(padded)

    encode_emotions = {
        "empty": 0,
        "sadness": 1,
        "enthusiasm": 2,
        "neutral": 3,
        "worry": 4,
        "surprise": 5,
        "love": 6,
        "fun": 7,
        "hate": 8,
        "happiness": 9,
        "boredom": 10,
        "relief": 11,
        "anger": 12,
    }

    predicted_class_label = np.argmax(model.predict(testing_padded), axis=-1)
    print(predicted_class_label)
    emotion = ""
    for key, value in encode_emotions.items():
        if value == predicted_class_label:
            emotion = key
            st.write("Prediction: ", key.upper())

    if st.button("Save Entry"):
        save(text_input, emotion)


st.write("Your Entries!")

day_entry_list = pd.read_csv("data_entry.csv")

col1, col2, col3 = st.columns(3)
for i in range(len(day_entry_list)):
    if i<3:
        date = day_entry_list.loc[i, "date"]
        text = day_entry_list.loc[i, "text"]
        emotion = day_entry_list.loc[i, "emotion"]
        st.write(date, text, emotion)


