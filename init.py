import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import ast

# Assuming songreccomendation.py has the following functions and objects: 
# emotion_mapping, knn, scaler, data, unique_emotions

# Set page configuration
st.set_page_config(layout="wide")

# Custom CSS to increase the width
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        max-width: 95%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the dataset and model from songreccomendation.py (assumed already run)
from songreccomendation import emotion_mapping, knn, scaler, data, unique_emotions

# Streamlit app setup
st.title("Music Recommendation System")

# Mood mapping (simplified for the UI; can expand as needed)
mood_mapping = {
    'Happy': {'danceability': 0.8, 'energy': 0.9, 'valence': 0.9, 'tempo': 120, 'genre': 'pop'},
    'Sad': {'danceability': 0.3, 'energy': 0.4, 'valence': 0.2, 'tempo': 90, 'genre': 'indie'},
    'Energetic': {'danceability': 0.9, 'energy': 0.95, 'valence': 0.8, 'tempo': 140, 'genre': 'dance'},
    'Relaxed': {'danceability': 0.5, 'energy': 0.4, 'valence': 0.6, 'tempo': 100, 'genre': 'r&b'},
    'Romantic': {'danceability': 0.7, 'energy': 0.6, 'valence': 0.8, 'tempo': 110, 'genre': 'pop'},
    'Angry': {'danceability': 0.6, 'energy': 0.95, 'valence': 0.3, 'tempo': 130, 'genre': 'rock'},
    'Confident': {'danceability': 0.7, 'energy': 0.8, 'valence': 0.7, 'tempo': 115, 'genre': 'pop rock'},
    'Melancholic': {'danceability': 0.4, 'energy': 0.3, 'valence': 0.2, 'tempo': 85, 'genre': 'blues'},
    'Optimistic': {'danceability': 0.8, 'energy': 0.7, 'valence': 0.9, 'tempo': 125, 'genre': 'indie pop'},
    'Pensive': {'danceability': 0.5, 'energy': 0.5, 'valence': 0.4, 'tempo': 95, 'genre': 'folk'},
    'Adventurous': {'danceability': 0.7, 'energy': 0.8, 'valence': 0.6, 'tempo': 130, 'genre': 'rock'},
    'Playful': {'danceability': 0.9, 'energy': 0.9, 'valence': 0.8, 'tempo': 135, 'genre': 'funk'},
    'Nostalgic': {'danceability': 0.6, 'energy': 0.5, 'valence': 0.7, 'tempo': 100, 'genre': 'shoegaze'},
    'Intense': {'danceability': 0.5, 'energy': 0.95, 'valence': 0.3, 'tempo': 140, 'genre': 'metal'},
    'Soothing': {'danceability': 0.4, 'energy': 0.3, 'valence': 0.5, 'tempo': 85, 'genre': 'ambient'},
    'Dreamy': {'danceability': 0.6, 'energy': 0.5, 'valence': 0.8, 'tempo': 95, 'genre': 'dream pop'},
    'Uplifted': {'danceability': 0.8, 'energy': 0.7, 'valence': 0.9, 'tempo': 125, 'genre': 'dance pop'},
    'Dramatic': {'danceability': 0.5, 'energy': 0.8, 'valence': 0.4, 'tempo': 110, 'genre': 'orchestral'}
}

# Zodiac to emotions mapping
zodiac_mapping = {
    "aries": ["energetic", "bold", "adventurous", "confident"],
    "taurus": ["grounded", "sensual", "reliable", "calm"],
    "gemini": ["curious", "dynamic", "adaptable", "communicative"],
    "cancer": ["emotional", "intuitive", "nurturing", "protective"],
    "leo": ["charismatic", "confident", "dramatic", "creative"],
    "virgo": ["analytical", "practical", "detail-oriented", "modest"],
    "libra": ["balanced", "charming", "diplomatic", "harmonious"],
    "scorpio": ["intense", "passionate", "mysterious", "determined"],
    "sagittarius": ["optimistic", "adventurous", "independent", "philosophical"],
    "capricorn": ["disciplined", "ambitious", "responsible", "practical"],
    "aquarius": ["innovative", "unique", "intellectual", "humanitarian"],
    "pisces": ["empathetic", "imaginative", "sensitive", "compassionate"]
}

# MBTI to emotions mapping
mbti_mapping = {
    "istj": ["responsible", "reliable", "practical", "systematic"],
    "isfj": ["dedicated", "warm", "caring", "detailed"],
    "infj": ["visionary", "insightful", "compassionate", "idealistic"],
    "intj": ["strategic", "independent", "analytical", "determined"],
    "istp": ["adventurous", "practical", "analytical", "resourceful"],
    "isfp": ["creative", "gentle", "spontaneous", "adaptable"],
    "infp": ["idealistic", "empathic", "creative", "curious"],
    "intp": ["innovative", "logical", "abstract", "independent"],
    "estp": ["energetic", "practical", "spontaneous", "action-oriented"],
    "esfp": ["outgoing", "enthusiastic", "friendly", "playful"],
    "enfp": ["inspirational", "creative", "curious", "sociable"],
    "entp": ["inventive", "dynamic", "quick-witted", "charismatic"],
    "estj": ["efficient", "organized", "decisive", "practical"],
    "esfj": ["supportive", "sociable", "empathetic", "cooperative"],
    "enfj": ["charismatic", "empathetic", "persuasive", "organized"],
    "entj": ["assertive", "strategic", "ambitious", "decisive"]
}

# Create the input UI for user mood, genre, zodiac sign, and MBTI type
st.write("Choose your mood and optional genre, zodiac sign, and MBTI type for music recommendations")

selected_mood = st.selectbox(
    "Select your mood:",
    list(mood_mapping.keys())
)

# Optionally allow the user to input a genre
genre_input = st.text_input("Optional: Enter a genre (leave blank to skip)")

# Optionally allow the user to input a zodiac sign
zodiac_input = st.selectbox(
    "Optional: Select your zodiac sign:",
    list(zodiac_mapping.keys()) + ['None']
)

# Optionally allow the user to input an MBTI type
mbti_input = st.selectbox(
    "Optional: Select your MBTI type:",
    list(mbti_mapping.keys()) + ['None']
)

# Fetch mood features
mood_features = mood_mapping[selected_mood]
st.write(f"Mapped features for mood '{selected_mood}':")

# Prepare input feature array
input_features = np.array([[mood_features['danceability'], mood_features['energy'], 
                            mood_features['valence'], mood_features['tempo']]])

# Function to recommend songs
def recommend_songs(knn_model, scaler, input_features, genre=None, zodiac=None, mbti=None):
    emotion_vector = np.zeros(len(unique_emotions))  # Initialize zero vector for emotions
    
    # Map genre to emotions
    if genre and genre.lower() in emotion_mapping:
        genre_emotions = emotion_mapping[genre.lower()]
        for emotion in genre_emotions:
            if emotion in unique_emotions:
                idx = list(unique_emotions).index(emotion)
                emotion_vector[idx] = 1

    # Map zodiac to emotions
    if zodiac and zodiac.lower() in zodiac_mapping:
        zodiac_emotions = zodiac_mapping[zodiac.lower()]
        for emotion in zodiac_emotions:
            if emotion in unique_emotions:
                idx = list(unique_emotions).index(emotion)
                emotion_vector[idx] = 1

    # Map MBTI to emotions
    if mbti and mbti.lower() in mbti_mapping:
        mbti_emotions = mbti_mapping[mbti.lower()]
        for emotion in mbti_emotions:
            if emotion in unique_emotions:
                idx = list(unique_emotions).index(emotion)
                emotion_vector[idx] = 1

    # Combine the user input features and emotion features
    combined_features = np.concatenate((input_features, emotion_vector.reshape(1, -1)), axis=1)
    combined_features_scaled = scaler.transform(combined_features)
    
    # Get the nearest neighbors
    distances, indices = knn_model.kneighbors(combined_features_scaled)
    recommended_songs = []
    printed_tracks = set()  # To track and avoid duplicates

    for index in indices[0]:
        track_name = data.iloc[index]['Track Name']
        artist_name = data.iloc[index]['Artist Name']
        genres = ', '.join(data.iloc[index]['Genres'])
        
        if track_name not in printed_tracks:
            recommended_songs.append(f"Track: {track_name}, Artist: {artist_name}, Genres: {genres}")
            printed_tracks.add(track_name)
    
    return recommended_songs

# Recommend songs based on mood, genre, zodiac, and MBTI
if st.button("Recommend Songs"):
    recommendations = recommend_songs(knn, scaler, input_features, genre_input, zodiac_input if zodiac_input != 'None' else None, mbti_input if mbti_input != 'None' else None)
    
    if recommendations:
        st.header("Recommended Songs:")
        for song in recommendations:
            st.write(song)
    else:
        st.write("No recommendations found for the selected mood, genre, zodiac, and MBTI.")