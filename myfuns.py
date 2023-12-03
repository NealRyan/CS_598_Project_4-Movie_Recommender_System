from myfuns import (genres, get_displayed_movies, get_popular_movies,
                    get_recommended_movies)

import os
import pandas as pd
import numpy as np

def genres():
    movies_data_filename = "Data/movies.dat"

    #Movies
    movies = pd.read_csv(movies_data_filename, sep='::', engine = 'python',
                        encoding="ISO-8859-1", header = None)
    movies.columns = ['MovieID', 'Title', 'Genres']
    multiple_idx = pd.Series([("|" in movie) for movie in movies['Genres']])
    movies.loc[multiple_idx, 'Genres'] = 'Multiple'


    genres = movies['Genres'].unique()
    return genres


def get_displayed_movies(num_to_return=10):
    movies_data_filename = "Data/movies.dat"
    ratings_data_filename = "Data/ratings.dat"
    users_data_filename = "Data/users.dat"

    #Movies
    movies = pd.read_csv(movies_data_filename, sep='::', engine = 'python',
                        encoding="ISO-8859-1", header = None)
    movies.columns = ['MovieID', 'Title', 'Genres']
    detailed_movies = movies.copy()
    multiple_idx = pd.Series([("|" in movie) for movie in movies['Genres']])
    movies.loc[multiple_idx, 'Genres'] = 'Multiple'

    #Ratings
    ratings = pd.read_csv(ratings_data_filename, sep='::', engine = 'python', header=None)
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings = ratings.drop('Timestamp', axis = 1)

    #Users
    users = pd.read_csv(users_data_filename, sep='::', engine = 'python', header=None)
    users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

    user_avg_ratings = ratings.groupby('UserID')['Rating'].mean().reset_index()
    user_avg_ratings.columns = ['UserID', 'Avg_User_Rating']

    ratings = pd.merge(ratings, user_avg_ratings, on='UserID')
    ratings['Normalized_Rating'] = ratings['Rating'] / ratings['Avg_User_Rating']

    genre_dummies = detailed_movies['Genres'].str.get_dummies(sep='|')
    detailed_movies = pd.concat([detailed_movies, genre_dummies], axis=1)

    movie_avg_ratings = ratings.groupby('MovieID').agg({'Rating': ['mean', 'count'], 'Normalized_Rating': 'mean'}).reset_index()
    movie_avg_ratings.columns = ['MovieID', 'Avg_Rating', 'Num_Ratings', 'Avg_Normalized_Rating']
    movies_with_ratings = pd.merge(detailed_movies, movie_avg_ratings, on='MovieID', how='left')
    movies_with_ratings.dropna(inplace=True)
    movies_with_ratings.sort_values(by='Avg_Normalized_Rating', ascending=False)

    weight_normalized_rating = 0.9 
    weight_num_ratings = 1-weight_normalized_rating
    max_num_ratings = 1000

    movies_with_ratings['Capped_Num_Ratings'] = movies_with_ratings['Num_Ratings'].clip(upper=max_num_ratings)

    movies_with_ratings['Score'] = (
        weight_normalized_rating * movies_with_ratings['Avg_Normalized_Rating'] +
        weight_num_ratings * movies_with_ratings['Capped_Num_Ratings']
    )

    movies_with_ratings.drop('Capped_Num_Ratings', axis=1, inplace=True)

    return movies_with_ratings.sort_values(by='Score', ascending=False).head(num_to_return)
    


def get_popular_movies(genre):
    
    movies_data_filename = "Data/movies.dat"
    ratings_data_filename = "Data/ratings.dat"
    users_data_filename = "Data/users.dat"

    #Movies
    movies = pd.read_csv(movies_data_filename, sep='::', engine = 'python',
                        encoding="ISO-8859-1", header = None)
    movies.columns = ['MovieID', 'Title', 'Genres']
    detailed_movies = movies.copy()
    multiple_idx = pd.Series([("|" in movie) for movie in movies['Genres']])
    movies.loc[multiple_idx, 'Genres'] = 'Multiple'

    #Ratings
    ratings = pd.read_csv(ratings_data_filename, sep='::', engine = 'python', header=None)
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings = ratings.drop('Timestamp', axis = 1)

    #Users
    users = pd.read_csv(users_data_filename, sep='::', engine = 'python', header=None)
    users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

    user_avg_ratings = ratings.groupby('UserID')['Rating'].mean().reset_index()
    user_avg_ratings.columns = ['UserID', 'Avg_User_Rating']

    ratings = pd.merge(ratings, user_avg_ratings, on='UserID')
    ratings['Normalized_Rating'] = ratings['Rating'] / ratings['Avg_User_Rating']

    genre_dummies = detailed_movies['Genres'].str.get_dummies(sep='|')
    detailed_movies = pd.concat([detailed_movies, genre_dummies], axis=1)

    movie_avg_ratings = ratings.groupby('MovieID').agg({'Rating': ['mean', 'count'], 'Normalized_Rating': 'mean'}).reset_index()
    movie_avg_ratings.columns = ['MovieID', 'Avg_Rating', 'Num_Ratings', 'Avg_Normalized_Rating']
    movies_with_ratings = pd.merge(detailed_movies, movie_avg_ratings, on='MovieID', how='left')
    movies_with_ratings.dropna(inplace=True)
    movies_with_ratings.sort_values(by='Avg_Normalized_Rating', ascending=False)

    weight_normalized_rating = 0.9 
    weight_num_ratings = 1-weight_normalized_rating
    max_num_ratings = 1000

    movies_with_ratings['Capped_Num_Ratings'] = movies_with_ratings['Num_Ratings'].clip(upper=max_num_ratings)

    movies_with_ratings['Score'] = (
        weight_normalized_rating * movies_with_ratings['Avg_Normalized_Rating'] +
        weight_num_ratings * movies_with_ratings['Capped_Num_Ratings']
    )

    movies_with_ratings.drop('Capped_Num_Ratings', axis=1, inplace=True)

    movies_with_ratings.sort_values(by='Score', ascending=False)

    genres = movies['Genres'].unique()
    return top_movies_by_genre(genre, df=movies_with_ratings)


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
    sorted_genre_movies = sorted_genre_movies[~sorted_genre_movies['MovieID'].isin(current_recommendations_df['MovieID'])]
    top_genre_movies = sorted_genre_movies.head(num_to_pad)[['MovieID', 'Title', metric]]
    
    return top_genre_movies


def top_movies_by_genre(genre, df, metric='Score', num=10):
    #Recommend the top num scoring movies from the genre specified
    
    genre_filter = df[genre] == 1
    genre_movies = df[genre_filter]
    sorted_genre_movies = genre_movies.sort_values(by=metric, ascending=False)
    top_genre_movies = sorted_genre_movies.head(num)[['MovieID', 'Title', metric]]
    top_genre_movies = pad_recommendations(top_genre_movies)

    return top_genre_movies