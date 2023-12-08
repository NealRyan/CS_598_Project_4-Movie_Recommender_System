#functions to implement
#DONE: genres, get_displayed_movies, get_popular_movies,
#TO DO: get_recommended_movies

import pandas as pd
import requests

#Expected movies column names:
#movie_id, title, genres
'''
# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)
'''

##########################
# Movies with ratings DF #
##########################

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

#########################
## Necessary Functions ##
#########################

def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(new_user_ratings):
    return movies.head(10)

def get_popular_movies(genre: str):
    if genre in genres:
        return top_movies_by_genre(genre, df=movies_with_ratings)
    else: 
        return movies[10:20]

