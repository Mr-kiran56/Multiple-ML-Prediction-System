from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import streamlit as st
import difflib
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from streamlit_option_menu import option_menu  
import re
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import io

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "🧠 Multiple Prediction System",
        ["📚 Books Recomendation", "📰 FakeNews Prediction", "✍️ Handwritten Digit Predition"]
    )

# Load models and data
book_data = pd.read_csv("B:\Streanlit\New folder\goodreads_data.csv")
regressor = pickle.load(open("B:/Streanlit/New folder/newnewspredict.sav", 'rb'))
tfidf_vectorizer = pickle.load(open("B:/Streanlit/New folder/tdfvecto.pkl", 'rb'))
model = pickle.load(open("B:/Streanlit/New folder/minitsmodel.sav", 'rb'))

# 📚 Book Recommendation System
if selected == "📚 Books Recomendation":
    st.title('📚 Book Recommendation by ML')
    data = book_data.fillna('')
    combined_data = data['Author'] + ' ' + data['Book'] + ' ' + data['Description'] + ' ' + data['Genres']
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_data)
    similarity = cosine_similarity(tfidf_matrix)

    input_book = st.text_input("Enter your preferred book:")

    # Clean genres
    book_data['Genres'] = book_data['Genres'].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", "").strip())
    all_genres = []
    for genre_string in book_data['Genres']:
        genres = [g.strip() for g in genre_string.split(',')]
        all_genres.extend(genres)
    unique_genres = list(set(all_genres))

    combined_values = (
        list(book_data['Book'].astype(str)) +
        list(book_data['Author'].astype(str)) +
        list(map(str, unique_genres)) +
        list(book_data['Description'].astype(str))
    )

    matches = difflib.get_close_matches(input_book, combined_values, n=5, cutoff=0.5)

    if matches:
        closest_match = matches[0]

        filtered_index = book_data[
            (book_data['Book'] == closest_match) |
            (book_data['Author'] == closest_match) |
            (book_data['Genres'].str.contains(closest_match, case=False)) |
            (book_data['Description'].str.contains(closest_match, case=False))
        ].index

        if not filtered_index.empty:
            index_of_book = filtered_index[0]
            similarity_scores = list(enumerate(similarity[index_of_book]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            if st.button("🔍 Find Recommendations"):
                st.subheader("Top 10 Book Recommendations:")
                for i in similarity_scores[1:11]:
                    index = i[0]
                    title = book_data.iloc[index]['Book']
                    author = book_data.iloc[index]['Author']
                    st.success(f"👉 {title} — by {author}")
        else:
            st.warning(f"No close match found for: {input_book}")
    elif input_book:
        st.warning("No similar books found. Try a different title.")

# 📰 Fake News Prediction
if selected == "📰 FakeNews Prediction":
    st.title('📰 Fake News Prediction by ML')
    nltk.download('stopwords')
    inputs = st.text_input('Enter news:')

    portter_stem = PorterStemmer()
    english_stopwords = set(stopwords.words('english'))

    def stemming_word(context):
        stemed_word = re.sub('[^a-zA-Z]', ' ', context)
        stemed_word = stemed_word.lower()
        stemed_word = stemed_word.split()
        stemed_word = [portter_stem.stem(word) for word in stemed_word if word not in english_stopwords]
        return ' '.join(stemed_word)

    if inputs:
        processed_input = stemming_word(inputs)
        X = tfidf_vectorizer.transform([processed_input])
        if st.button('Predict News'):
            x_testpredictinput = regressor.predict(X)
            if x_testpredictinput == 0:
                st.success('✅ It is Real News')
                st.markdown("### 🟢 Verified ✔️")
            else:
                st.error('❌ It is Fake News')
                st.markdown("### 🔴 FAKE ❗📰🚫")

# ✍️ Handwritten Digit Prediction
if selected == "✍️ Handwritten Digit Predition":
    st.title("✍️ Handwritten Digit Recognition")
    uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if img is None:
            st.error("Image not found or unreadable.")
        else:
            img_resized = cv2.resize(img, (28, 28))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_gray = img_gray / 255.0
            img_input = np.reshape(img_gray, (1, 28, 28))

            prediction = model.predict(img_input)
            predicted_digit = np.argmax(prediction)

            st.image(img_gray, caption=f"Predicted: {predicted_digit}", width=150, channels="GRAY")
            st.success(f"🧠 Predicted Digit: {predicted_digit}")
