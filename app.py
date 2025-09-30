import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = {
    "title": [
        "The Dark Knight",
        "Inception",
        "Interstellar",
        "Avengers: Endgame",
        "Iron Man",
        "The Matrix",
        "Titanic",
        "The Shawshank Redemption",
        "The Godfather",
        "Spider-Man: No Way Home"
    ],
    "genre": [
        "Action Crime Drama",
        "Action Sci-Fi Thriller",
        "Adventure Drama Sci-Fi",
        "Action Adventure Sci-Fi",
        "Action Sci-Fi",
        "Action Sci-Fi",
        "Romance Drama",
        "Drama Crime",
        "Crime Drama",
        "Action Sci-Fi Adventure"
    ]
}

df = pd.DataFrame(movies)

# Vectorization
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['genre'])

# Similarity
cosine_sim = cosine_similarity(count_matrix)

# Recommendation function
def recommend_movie(movie_title):
    if movie_title not in df['title'].values:
        return ["Movie not found in database. Try another."]
    movie_index = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]  # top 5
    recommended_movies = [df.iloc[i[0]]['title'] for i in sorted_scores]
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")

st.write("Type a movie name and get similar recommendations!")

movie_input = st.selectbox("Choose a movie", df['title'].values)

if st.button("Recommend"):
    results = recommend_movie(movie_input)
    st.subheader("Recommended Movies:")
    for r in results:
        st.write(f"👉 {r}")

# Show dataset option
if st.checkbox("Show Movie Dataset"):
    st.dataframe(df)
