import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Define column names for the datasets
ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
movies_columns = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# Load the data
ratings = pd.read_csv('u.data', sep='\t', names=ratings_columns)
movies = pd.read_csv('u.item', sep='|', names=movies_columns, encoding='latin-1')

# Merge ratings and movies dataframes
data = pd.merge(ratings, movies, on='item_id')

# Create a user-item matrix
user_item_matrix = data.pivot_table(index='user_id', columns='title', values='rating')
user_item_matrix.fillna(0, inplace=True)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to recommend movies based on user similarity
def get_user_recommendations(user_id, user_similarity_df, user_item_matrix, top_n=10):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    similar_users_ratings = user_item_matrix.loc[similar_users]
    weighted_sum_ratings = similar_users_ratings.mul(user_similarity_df[user_id].sort_values(ascending=False)[1:], axis=0).sum(axis=0)
    sum_similarity_scores = user_similarity_df[user_id].sort_values(ascending=False)[1:].sum()
    predicted_ratings = weighted_sum_ratings / sum_similarity_scores
    unrated_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 0]
    recommendations = predicted_ratings.loc[unrated_movies.index]
    return recommendations.sort_values(ascending=False).head(top_n)

# Streamlit app
st.title("Movie Recommendation System")

# User input for user ID
user_id = st.number_input("Enter User ID:", min_value=1, max_value=ratings['user_id'].max())

if st.button("Recommend"):
    recommended_movies = get_user_recommendations(user_id, user_similarity_df, user_item_matrix)
    st.write("Recommended Movies for User {}:".format(user_id))
    for movie in recommended_movies.index:
        st.write(movie)
