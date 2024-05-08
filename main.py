import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample movie dataset (replace this with your dataset)
movies = pd.DataFrame({
    'Title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
    'Genre': ['Drama', 'Crime', 'Action', 'Crime', 'Drama'],
    'Director': ['Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan', 'Quentin Tarantino', 'Robert Zemeckis']
})


# Function to recommend movies based on similarity
def recommend_movies(movie_title, num_recommendations=5):
    # Create a CountVectorizer to convert text data into matrix format
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(movies['Genre'] + " " + movies['Director'])

    # Compute cosine similarity matrix based on the count matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Get the index of the movie that matches the title
    idx = movies[movies['Title'] == movie_title].index[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the num_recommendations most similar movies
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top num_recommendations most similar movies
    return movies['Title'].iloc[movie_indices]


# Example usage
if __name__ == "__main__":
    movie_title = 'The Godfather'
    recommendations = recommend_movies(movie_title)
    print(f"Recommended movies similar to '{movie_title}':")
    for movie in recommendations:
        print(movie)
