import numpy as np
import pandas as pd
import csv
from PIL import Image

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

emo_code_url = {
    "empty": [0, "./emoticons/Empty.png"],
    "sadness": [1, "./emoticons/Sadness.png"],
    "enthusiasm": [2, "./emoticons/Enthusiasm.png"],
    "neutral": [3, "./emoticons/Neutral.png"],
    "worry": [4, "./emoticons/Worry.png"],
    "surprise": [5, "./emoticons/Surprise.png"],
    "love": [6, "./emoticons/Love.png"],
    "fun": [7, "./emoticons/Fun.png"],
    "hate": [8, "./emoticons/Hate.png"],
    "happiness": [9, "./emoticons/Happiness.png"],
    "boredom": [10, "./emoticons/Boredom.png"],
    "relief": [11, "./emoticons/Relief.png"],
    "anger": [12, "./emoticons/Anger.png"],
}


def save(text, emotion):
    with open("data_entry.csv", "a") as f:
        f.write("%s,%s,%s\n" % (date_time, text, emotion))


def app_headers():
    # Title
    st.title("Digital Journal")
    # Day and Date
    st.write(date_time)


def new_entry():
    # New Entry
    input = st.empty()
    text = str(input.text_input("How was your day?"))

    if text != "":
        sentence = []
        sentence.append(text)
        print(sentence)

        sequences = tokenizer.texts_to_sequences(sentence)
        print("sequence", sequences)

        padded = pad_sequences(
            sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
        )
        testing_padded = np.array(padded)

        predicted_class_label = np.argmax(model.predict(testing_padded), axis=-1)
        print(predicted_class_label)
        emotion = ""
        emoticon = ""
        col1, col2, col3 = st.columns(3)
        for key, value in emo_code_url.items():
            if value[0] == predicted_class_label:
                emotion = key
                emoticon = value[1]
                with col1:
                    st.write(key.upper())
                with col2:
                    image = Image.open(emoticon)
                    st.image(image, width=60)
        
        with col3:
            if st.button("Save Entry"):                              
                save(text, emotion)
                # text = input.text_input("") 
        
        


def display_entries():

    st.header("Your Entries!")

    day_entry_list = pd.read_csv("data_entry.csv")
    day_entry_list["date"] = pd.to_datetime(day_entry_list["date"])
    day_entry_list = day_entry_list.sort_values(
        by="date", ascending=False, ignore_index=True
    )

    # st.write(day_entry_list)

    col1, col2, col3 = st.columns(3)

    for i in range(len(day_entry_list)):

        if i < 3:

            date = day_entry_list.loc[i, "date"]
            text = day_entry_list.loc[i, "text"]
            emotion = day_entry_list.loc[i, "emotion"]
            image = ""

            for key, value in emo_code_url.items():
                
                if emotion == key:
                    print(value[1])
                    image = Image.open(value[1])

            if i + 1 == 1:
                with col1:
                    st.subheader(date.strftime("%A, %d %B %Y"))
                    st.write(text)
                    st.write(emotion.upper())
                    st.image(image, width=60)
            if i + 1 == 2:
                with col2:
                    st.subheader(date.strftime("%A, %d %B %Y"))
                    st.write(text)
                    st.write(emotion.upper())
                    st.image(image, width=60)
            if i + 1 == 3:
                with col3:                   
                    st.subheader(date.strftime("%A, %d %B %Y"))
                    st.write(text)
                    st.write(emotion.upper())
                    st.image(image, width=60)


## Calling methods......

app_headers()
new_entry()
display_entries()
