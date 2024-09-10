
import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load the cleaned genres dataset
genres_df = pd.read_csv('cleaned_genres.csv')

import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load the cleaned genres dataset
genres_df = pd.read_csv('cleaned_genres.csv')

# Expand the genre to emotion mapping
emotion_mapping = {
    # Indie genres
    "indie psych-pop": ["dreamy", "uplifting", "psychedelic", "hopeful"],
    "indie psychedelic rock": ["trippy", "adventurous", "surreal", "free-spirited"],
    "indie punk": ["rebellious", "energetic", "angsty", "defiant"],
    "indie quebecois": ["introspective", "moody", "melancholic", "artistic"],
    "indie r&b": ["sensual", "soulful", "smooth", "romantic"],
    "indie rock": ["youthful", "exploratory", "confident", "laid-back"],
    "indie shoegaze": ["ethereal", "nostalgic", "dreamlike", "distant"],
    
    # Indonesian genres
    "indonesian blues": ["soulful", "reflective", "melancholy", "warm"],
    "indonesian electronic": ["energized", "hypnotic", "futuristic", "immersive"],
    "indonesian experimental": ["abstract", "unpredictable", "unconventional", "thought-provoking"],
    "indonesian hip hop": ["gritty", "motivational", "bold", "authentic"],
    "indonesian indie": ["nostalgic", "dreamy", "relaxed", "introspective"],
    
    # Industrial genres
    "industrial black metal": ["intense", "dark", "aggressive", "raw"],
    "industrial hip hop": ["edgy", "rebellious", "mechanical", "futuristic"],
    "industrial metal": ["powerful", "chaotic", "energetic", "intense"],
    
    # Instrumental genres
    "instrumental bluegrass": ["optimistic", "uplifting", "joyful", "earthy"],
    "instrumental death metal": ["dark", "complex", "aggressive", "melancholic"],
    "instrumental funk": ["groovy", "playful", "upbeat", "cool"],
    "instrumental post-rock": ["reflective", "expansive", "cinematic", "meditative"],
    
    # Irish genres
    "irish black metal": ["dark", "intense", "mysterious", "melancholic"],
    "irish folk": ["warm", "joyful", "storytelling", "nostalgic"],
    "irish indie rock": ["melancholic", "youthful", "free-spirited", "contemplative"],
    
    # Italian genres
    "italian hip hop": ["energetic", "bold", "motivational", "streetwise"],
    "italian indie pop": ["dreamy", "romantic", "melancholic", "youthful"],
    "italian gothic metal": ["dark", "dramatic", "haunting", "intense"],
    "italian pop": ["vibrant", "romantic", "uplifting", "joyful"],
    
    # J-pop and K-pop
    "j-pop": ["energetic", "colorful", "cheerful", "youthful"],
    "j-rock": ["dynamic", "intense", "empowering", "youthful"],
    "k-pop": ["catchy", "energetic", "bright", "cheerful"],
    "k-rap": ["bold", "confident", "motivational", "streetwise"],
    
    # Latin genres
    "latin pop": ["romantic", "passionate", "uplifting", "vibrant"],
    "latin rock": ["intense", "energetic", "rebellious", "passionate"],
    "latin jazz": ["sophisticated", "smooth", "soulful", "relaxed"],
    
    # Metal genres
    "melodic death metal": ["intense", "melancholic", "complex", "emotional"],
    "melodic metalcore": ["aggressive", "empowering", "dramatic", "bold"],
    "progressive metal": ["complex", "introspective", "expansive", "cerebral"],
    
    # Neo-soul and R&B
    "neo soul": ["soulful", "romantic", "smooth", "reflective"],
    "r&b": ["sensual", "romantic", "smooth", "intimate"],
    
    # Electronic genres
    "synthwave": ["nostalgic", "futuristic", "dreamy", "energizing"],
    "tech house": ["groovy", "hypnotic", "steady", "energetic"],
    "trance": ["uplifting", "euphoric", "dreamlike", "energized"],
    
    # Rock and Punk
    "punk rock": ["rebellious", "raw", "energetic", "angsty"],
    "classic rock": ["nostalgic", "powerful", "free-spirited", "empowering"],
    "progressive rock": ["introspective", "complex", "expansive", "dreamy"],
    
    # Blues and Jazz
    "blues": ["soulful", "melancholic", "reflective", "warm"],
    "jazz": ["sophisticated", "relaxed", "smooth", "improvisational"],
    "jazz fusion": ["dynamic", "complex", "innovative", "cerebral"],
    
    # Other genres
    "folk": ["earthy", "nostalgic", "storytelling", "heartwarming"],
    "lo-fi beats": ["chill", "relaxed", "introspective", "soothing"],
    "world music": ["vibrant", "cultural", "expansive", "dynamic"],
    
    # Hip hop and Rap
    "hip hop": ["bold", "motivational", "streetwise", "intense"],
    "trap": ["edgy", "dark", "gritty", "empowering"],
    "conscious rap": ["thought-provoking", "introspective", "serious", "emotive"],
    
    # Pop
    "pop": ["catchy", "uplifting", "bright", "cheerful"],
    "dream pop": ["ethereal", "dreamlike", "introspective", "romantic"],
    "pop rock": ["youthful", "empowering", "vibrant", "feel-good"],
    
    # Classical and Orchestral
    "neo-classical": ["reflective", "emotional", "melancholic", "expansive"],
    "orchestral": ["dramatic", "epic", "emotional", "cinematic"]
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

# Load additional genres and update mapping
additional_genres = pd.read_csv('cleaned_genres.csv')
for genre in additional_genres:
    if genre not in emotion_mapping:
        emotion_mapping[genre] = ["varied emotions, alternative, branch genre"]

# Zodiac and MBTI mappings (as defined above)

# List of dataset filenames
files = [
    'alternative_music_data.csv', 
    'blues_music_data.csv', 
    'hiphop_music_data.csv',
    'indie_alt_music_data.csv',
    'metal_music_data.csv', 
    'pop_music_data.csv', 
    'rock_music_data.csv'
]

# Load each CSV file into a DataFrame and store in a list
dataframes = [pd.read_csv(file) for file in files]

# Combine all DataFrames into a single DataFrame
data = pd.concat(dataframes, ignore_index=True)

data.columns = data.columns.str.strip()
print(data.columns)
print([col for col in data.columns if 'genre' in col.lower()])

# Convert 'Genres' column from string representation of list to actual list
data['Genres'] = data['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Function to map genres to emotions
def map_genres_to_emotions(genres_list):
    emotions = []
    for genre in genres_list:
        if genre in emotion_mapping:
            emotions.extend(emotion_mapping[genre])
    return emotions

# Apply the mapping function to each row
data['Emotions'] = data['Genres'].apply(map_genres_to_emotions)

# Create binary features for each unique emotion across all songs
unique_emotions = set(emotion for sublist in data['Emotions'] for emotion in sublist)
for emotion in unique_emotions:
    data[emotion] = data['Emotions'].apply(lambda x: 1 if emotion in x else 0)

# Define the features including the numerical emotion features
numerical_features = ['danceability', 'energy', 'valence', 'tempo']
emotion_features = list(unique_emotions)

# Prepare the feature set for scaling and KNN
X = data[numerical_features + emotion_features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit the KNN model
knn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
knn.fit(X_scaled)

# Function to process user input and provide recommendations
def recommend_songs(knn_model, scaler, input_features, emotion_mapping, data, genre=None, zodiac=None, mbti=None):
    # Initialize emotion vector
    emotion_vector = np.zeros(len(unique_emotions))
    
    # Map genre to emotions
    if genre:
        if genre in emotion_mapping:
            genre_emotions = emotion_mapping[genre]
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

    # Scale the combined features using the existing scaler
    combined_features_scaled = scaler.transform(combined_features)

    # Get the nearest neighbors
    distances, indices = knn_model.kneighbors(combined_features_scaled)

    # Retrieve and print the recommended songs, ensuring uniqueness
    print("\nRecommended Songs:")
    printed_tracks = set()  # To track and avoid duplicates
    for index in indices[0]:
        track_name = data.iloc[index]['Track Name']
        artist_name = data.iloc[index]['Artist Name']
        genres = data.iloc[index]['Genres']
        
        if track_name not in printed_tracks:
            print(f"Track: {track_name}\nArtist: {artist_name}\nGenres: {genres}\n")
            printed_tracks.add(track_name)