# myfuns.py
# for Project 4, CS 598 Practical Statistical Learning, Fall 2023
#
# Core logic for movie recommendation app using the MovieLens dataset.
#
# Team Members:
# * Neal Ryan (nealpr2): System I + padding logic for System II
# * Kurt Tuohy (ktuohy): System II similarity matrix and core recommendation system
# * Alelign Faris (faris2): Python Dash app
#
# To run app locally:
#
# 1) From command prompt, run:
#    python Dash_Starter_Code.py
#
# 2) Point web browser to: http://127.0.0.1:8080/


import pandas as pd
import numpy as np

##########################
# Movies with ratings DF #
##########################

# Load list of movies in the MovieLens dataset

movies_data_filename = "Data/movies.dat"
ratings_data_filename = "Data/ratings.dat"
users_data_filename = "Data/users.dat"

#Movies
movies = pd.read_csv(movies_data_filename, sep='::', engine = 'python',
                    encoding="ISO-8859-1", header = None)
movies.columns = ['movie_id', 'title', 'genres']
detailed_movies = movies.copy()
multiple_idx = pd.Series([("|" in movie) for movie in movies['genres']])
movies.loc[multiple_idx, 'genres'] = 'Multiple'

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)


#Ratings
ratings = pd.read_csv(ratings_data_filename, sep='::', engine = 'python', header=None)
ratings.columns = ['UserID', 'movie_id', 'Rating', 'Timestamp']
ratings = ratings.drop('Timestamp', axis = 1)

#Users
users = pd.read_csv(users_data_filename, sep='::', engine = 'python', header=None)
users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

# Compute each MovieLens user's average rating.
# Use that to normalize ratings.

user_avg_ratings = ratings.groupby('UserID')['Rating'].mean().reset_index()
user_avg_ratings.columns = ['UserID', 'Avg_User_Rating']

ratings = pd.merge(ratings, user_avg_ratings, on='UserID')
ratings['Normalized_Rating'] = ratings['Rating'] / ratings['Avg_User_Rating']

genre_dummies = detailed_movies['genres'].str.get_dummies(sep='|')
detailed_movies = pd.concat([detailed_movies, genre_dummies], axis=1)

movie_avg_ratings = ratings.groupby('movie_id').agg({'Rating': ['mean', 'count'], 'Normalized_Rating': 'mean'}).reset_index()
movie_avg_ratings.columns = ['movie_id', 'Avg_Rating', 'Num_Ratings', 'Avg_Normalized_Rating']
movies_with_ratings = pd.merge(detailed_movies, movie_avg_ratings, on='movie_id', how='left')
movies_with_ratings.dropna(inplace=True)
movies_with_ratings.sort_values(by='Avg_Normalized_Rating', ascending=False)

# Compute weighted movie scores for Sytem I.
# Assign 90% of score to the average normalized rating, 10% to the winsorized number of ratings.

weight_normalized_rating = 0.9 
weight_num_ratings = 1-weight_normalized_rating
max_num_ratings = 1000

movies_with_ratings['Capped_Num_Ratings'] = movies_with_ratings['Num_Ratings'].clip(upper=max_num_ratings)

movies_with_ratings['Score'] = (
    weight_normalized_rating * movies_with_ratings['Avg_Normalized_Rating'] +
    weight_num_ratings * movies_with_ratings['Capped_Num_Ratings']
)

movies_with_ratings.drop('Capped_Num_Ratings', axis=1, inplace=True)

movies_with_ratings.sort_values(by='Score', ascending=False, inplace=True)

#################################
# Movie/movie similarity matrix #
#################################

# Read in the movie/movie similarity matrix we computed for System II.
similarity_matrix_filename = "Data/movie_similarity_matrix.csv"
similarity_df = pd.read_csv(similarity_matrix_filename, index_col=0)
similarity_matrix = similarity_df.to_numpy()

# The similarity matrix movie IDs are the same as in movies.dat, only prepended with "m".
char_movie_ids = similarity_df.columns


####################
# Helper Functions #
####################

def pad_recommendations(current_recommendations_df, num=10, genre='any', metric='Score'):
    #Pad any recommendation dataframe up to the specified number with the best 
    #movies from the genre (if specified, best in general if not)
    #Useful for sparse recommendations

    num_to_pad = num - len(current_recommendations_df)
    if num_to_pad==0:
        return current_recommendations_df
    
    if genre== 'any':
        pad_movies = current_recommendations_df
    else:
        genre_filter = current_recommendations_df[genre] == 1
        pad_movies = current_recommendations_df[genre_filter]
    sorted_genre_movies = pad_movies.sort_values(by=metric, ascending=False)
    sorted_genre_movies = sorted_genre_movies[~sorted_genre_movies['movie_id'].isin(current_recommendations_df['movie_id'])]
    top_genre_movies = sorted_genre_movies.head(num_to_pad)[['movie_id', 'title', metric]]
    #print(top_genre_movies.head())
    return top_genre_movies

def top_movies_by_genre(genre, df, metric='Score', num=10):
    #Recommend the top num scoring movies from the genre specified
    
    genre_filter = df[genre] == 1
    genre_movies = df[genre_filter]
    sorted_genre_movies = genre_movies.sort_values(by=metric, ascending=False)
    top_genre_movies = sorted_genre_movies.head(num)[['movie_id', 'title', metric]]
    top_genre_movies = pad_recommendations(top_genre_movies)

    return top_genre_movies

def construct_user_rating_array(user_rating_input):
    # Convert a dictionary of movie user ratings to a 1-D array for use in IBCF.
    # Each array element corresponds to a possible movie ID.

    # Initialize array
    user_rating_array = np.full((char_movie_ids.shape[0]), np.NAN)

    # Loop over user ratings. Set an individual array element for each rating.
    for movie_id, rating in user_rating_input.items():
        char_movie_id = "m" + str(movie_id)
        char_movie_index = np.where(char_movie_ids == char_movie_id)[0][0]
        user_rating_array[char_movie_index] = rating

    return user_rating_array

def top_alternate_movies(exclude_movie_ids, df=movies_with_ratings, metric="Score", num=10):
    """
    Recommend the top num high-scoring movies that are not in the given list.
    Use this to pad out similarity-based recommendations when there are too few.
    
    Arguments:
    * exclude_movie_ids: list of integer movie IDs that are already rated or recommended
    * df: dataframe of all movies, sorted by score
    * metric: a combined measure of movie popularity + rating.
    * num: the number of movies to return
    """
    
    # Exclude movies that are in the exclusion list
    sorted_movies = df[~df['movie_id'].isin(exclude_movie_ids)]
    # Take the top num movies
    top_num_movies = sorted_movies.head(num)[['movie_id', 'title']]

    return top_num_movies

#########################
## Necessary Functions ##
#########################

def get_displayed_movies():
    return movies_with_ratings[['movie_id', 'title', 'genres']].head(100)
    #return movies.head(100)

def get_popular_movies(genre: str):
    if genre in genres:
        return top_movies_by_genre(genre, df=movies_with_ratings)
    else: 
        return movies[10:20]

def get_recommended_movies(user_rating_input, n_req_recommendations=10, min_rating_recommended=3.0):
    """
    Given a viewer's movie ratings and a matrix of movie similarities,
    recommend different movies to the user.
    Recommend exactly n_req_recommendations movies.
    
    Arguments:
    * user_rating_input: dictionary of movie IDs and the user's ratings.
    * n_req_recommendations: the required number of recommendations to return.
    * min_rating_recommended: only recommend movies with this predicted rating and above.
    """

    # Convert rating dict to 1-D array, where each element correponds to potential movie.
    user_rating_array = construct_user_rating_array(user_rating_input)

    # Initialize return values
    pred_ratings_sorted = np.full((n_req_recommendations), np.NAN)
    # Recommended movies for user
    recommended_movie_ids = np.full((n_req_recommendations), np.NAN)
     
    # Get index values of movies the new user has rated.
    rated_index = np.squeeze(np.argwhere(np.logical_not(np.isnan(user_rating_array)))).tolist()
    
    # IDs of rated movies in the reduced matrix.
    # Convert IDs to integer for compatibility with the "movies" dataframe.

    # If only one rating
    if (type(rated_index) is int):
        rated_movie_ids = [int(char_movie_ids[rated_index][1:])]
    # If list of ratings
    elif (len(rated_index) > 1):
        rated_movie_ids = [int(movie_id[1:]) for movie_id in char_movie_ids[rated_index].tolist()]
    # If no ratings
    else:
        rated_movie_ids = []
    
    # Identify movies the new user hasn't rated yet.
    # Will use these to skip rating movies a user has already rated.
    unrated_index = np.squeeze(np.argwhere(np.isnan(user_rating_array))).tolist()

    # IDs of unrated movies in the reduced matrix
    unrated_movie_ids = char_movie_ids[unrated_index]
    
    # Initialize the number of recommendations to return
    n_recommended = 0

    # If user rated at least one movie, find similar movies to recommend
    if (len(unrated_index) < similarity_matrix.shape[0]):
    
        # To make predictions, create reduced version of the culled similarity matrix.
        # The user's rated movies will be columns and unrated movies will be rows.
        # Fill NaN values with zero to make calculations easier.

        S_for_pred = np.nan_to_num(similarity_matrix[unrated_index, :][:, rated_index], nan=0.0)

        # Numerators of predicted ratings
        pred_numerators = np.dot(S_for_pred, user_rating_array[rated_index])

        # Initialize predicted rating matrix.
        # Force NaNs to zero to help sort ratings.
        pred_ratings = np.full((pred_numerators.shape), fill_value=0.0, dtype="float64")

        # Get denominators of predicted ratings.

        # Similarity matrix for predictions is two-dimensional: normal case.
        if (S_for_pred.ndim > 1):
            pred_denominators = np.sum(S_for_pred, axis=1)
        # Similarity matrix is one-dimensional due to single rating
        else:
            pred_denominators = S_for_pred

        # Compute predicted ratings.

        # Indexes of nonzero denominators: movies with at least one similarity value to something the new user rated
        nonzero_denom_index = np.argwhere(pred_denominators > 0.0)
        # Generate predictions
        pred_ratings[nonzero_denom_index] = pred_numerators[nonzero_denom_index] / pred_denominators[nonzero_denom_index]

        # Count the predictions that are equal to or greater than min_rating_recommended.
        n_recommended = np.sum(pred_ratings >= min_rating_recommended)
        # Set a ceiling on the number of recommendations to return.
        n_recommended = min(n_recommended, n_req_recommendations)
        
    # If any movies recommended, sort them by descending rating
    if (n_recommended > 0):

        # Sort the predicted ratings, highest first. Return max n_req_recommendations ratings.
        pred_rating_sort_index = np.argsort(pred_ratings)[::-1][:n_recommended]
        # Sort movie IDs in the same order
        recommended_movie_ids = unrated_movie_ids[pred_rating_sort_index]
        # Convert recommended movie IDs to integers for compatibility with the "movies" dataframe
        recommended_movie_ids = [int(movie_id[1:]) for movie_id in recommended_movie_ids.tolist()]
        recommended_movie_id_df = pd.DataFrame({"movie_id": recommended_movie_ids})

        # Get dataframe of recommendations
        recommended_movies = pd.merge(recommended_movie_id_df, movies, on="movie_id")
        
        # If not enough recommendations, fill in the rest based on System I logic.

        if (n_recommended < n_req_recommendations):
            # Don't include existing ratings and recommendations in the list of additional movies.
            additional_movies = top_alternate_movies(exclude_movie_ids=rated_movie_ids + recommended_movie_ids,
                                                        df=movies_with_ratings,
                                                        num=n_req_recommendations - n_recommended)
            recommended_movies = pd.concat([recommended_movies, additional_movies])
            
    # Either user hasn't rated any movies, or no recommendations found. Use System I logic to make recommendations.
    # Exclude any existing ratings.
    else:
        recommended_movies = top_alternate_movies(exclude_movie_ids=rated_movie_ids, num=n_req_recommendations)

    # Return recommendations from the "movies" dataframe
    return recommended_movies
