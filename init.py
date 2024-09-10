import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import ast

# Assuming songreccomendation.py has the following functions and objects: 
# emotion_mapping, knn, scaler, data, unique_emotions

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
    'Romantic': {'danceability': 0.7, 'energy': 0.6, 'valence': 0.8, 'tempo': 110, 'genre': 'pop'}
}

# Create the input UI for user mood and optional genre
st.header("Choose your mood and optional genre for music recommendations")

selected_mood = st.selectbox(
    "Select your mood:",
    list(mood_mapping.keys())
)

# Optionally allow the user to input a genre
genre_input = st.text_input("Optional: Enter a genre (leave blank to skip)")

# Fetch mood features
mood_features = mood_mapping[selected_mood]
st.write(f"Mapped features for mood '{selected_mood}':")
st.write(f"Danceability: {mood_features['danceability']}, Energy: {mood_features['energy']}, "
         f"Valence: {mood_features['valence']}, Tempo: {mood_features['tempo']}")

# Prepare input feature array
input_features = np.array([[mood_features['danceability'], mood_features['energy'], 
                            mood_features['valence'], mood_features['tempo']]])

# Function to recommend songs
def recommend_songs(knn_model, scaler, input_features, genre=None):
    emotion_vector = np.zeros(len(unique_emotions))  # Initialize zero vector for emotions
    
    if genre and genre.lower() in emotion_mapping:
        genre_emotions = emotion_mapping[genre.lower()]
        # Set 1 in the corresponding emotion features
        for emotion in genre_emotions:
            if emotion in unique_emotions:
                idx = list(unique_emotions).index(emotion)
                emotion_vector[idx] = 1

    combined_features = np.concatenate((input_features, emotion_vector.reshape(1, -1)), axis=1)
    combined_features_scaled = scaler.transform(combined_features)
    
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

# Recommend songs based on mood and genre
if st.button("Recommend Songs"):
    recommendations = recommend_songs(knn, scaler, input_features, genre_input)
    
    if recommendations:
        st.header("Recommended Songs:")
        for song in recommendations:
            st.write(song)
    else:
        st.write("No recommendations found for the selected mood and genre.")