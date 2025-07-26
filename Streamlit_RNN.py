from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reversed_word_index = {v: k for k, v in word_index.items()}

model=load_model('imdb.h5')


def decode_review(encode_review):
    return ''.join([reversed_word_index.get(i-3, '?') for i in encode_review])

def preprocess_review(review):
    words = review.lower().split()
    encode_review= [word_index.get(word, 2) + 3 for word in words]  # +3 to account for reserved indices
    padding_review = sequence.pad_sequences([encode_review], maxlen=500)
    return padding_review



def predict_review(review):
    perprocessed_review = preprocess_review(review)
    prediction = model.predict(perprocessed_review)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]

#streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative):")

#user input
user_review = st.text_area("Review", "The movie was great!")

if st.button("Classify"):
    
        processed_input = preprocess_review(user_review)
        prediction = model.predict(processed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {prediction[0][0]:.4f}")

    

else:
    st.write("Please enter a movie review.")       

    
