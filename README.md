# Recommender_system
Movie recommendation system

# Import required libraries
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load data
ratings_data = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')

# Merge data frames
movie_ratings_data = pd.merge(ratings_data, movies_data, on='movieId')

# Create a pivot table of user ratings for each movie
movie_ratings_pivot = movie_ratings_data.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)

# Convert pivot table to sparse matrix format for efficient computation
movie_ratings_sparse = csr_matrix(movie_ratings_pivot)

# Build k-nearest neighbors model using cosine similarity
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_knn.fit(movie_ratings_sparse)

# Define function to recommend movies based on user ratings
def recommend_movies(movie_name):
  # Get index of movie in data frame
  movie_index = movies_data[movies_data['title'] == movie_name].index[0]

  # Get list of similar movies based on k-nearest neighbors model
  _, indices = model_knn.kneighbors(movie_ratings_pivot.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=10)

  # Print recommended movie titles
  print("Recommendations for", movie_name, ":")
  for i in range(1, len(indices.flatten())):
    print(i, "-", movies_data.iloc[indices.flatten()[i]]['title'])

# Test function with example movie
recommend_movies('Toy Story (1995)')

