import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from surprise import SVD, Dataset, Reader

from surprise.model_selection import accuracy



# Step 1: Load sample data (you can replace with real dataset)

data = {

    'user_id': [101, 102, 103, 104, 101],

    'movie_id': [201, 202, 203, 204, 201],

    'rating': [5, 3, 4, 2, 5]

}

df = pd.DataFrame(data)



# Step 2: Preprocess

df = df.drop_duplicates()

df = df.dropna()

df['rating'] = df['rating'].clip(1, 5)



# Encode user and movie IDs

user_enc = LabelEncoder()

movie_enc = LabelEncoder()

df['user_enc'] = user_enc.fit_transform(df['user_id'])

df['movie_enc'] = movie_enc.fit_transform(df['movie_id'])



# Step 3: Prepare data for Surprise library

reader = Reader(rating_scale=(1, 5))

data_surprise = Dataset.load_from_df(df[['user_enc', 'movie_enc', 'rating']], reader)



# Step 4: Train-test split

trainset, testset = train_test_split(df[['user_enc', 'movie_enc', 'rating']], test_size=0.2)



# Convert to surprise format

trainset_surprise = Dataset.load_from_df(trainset, reader).build_full_trainset()

testset_surprise = list(testset.itertuples(index=False, name=None))



# Step 5: Train SVD model

model = SVD()

model.fit(trainset_surprise)



# Step 6: Predict

predictions = model.test(testset_surprise)



# Step 7: Evaluate

rmse = accuracy.rmse(predictions)



# Step 8: Recommendation function

def recommend_movies(user_id, n=3):

    if user_id not in df['user_id'].values:

        print("User not found.")

        return

    

    user_idx = user_enc.transform([user_id])[0]

    all_movies = df['movie_enc'].unique()

    watched = df[df['user_enc'] == user_idx]['movie_enc'].values

    not_watched = [m for m in all_movies if m not in watched]

    

    pred_ratings = [(m, model.predict(user_idx, m).est) for m in not_watched]

    top_movies = sorted(pred_ratings, key=lambda x: x[1], reverse=True)[:n]

    

    print(f"Top {n} recommendations for User {user_id}:")

    for movie_idx, score in top_movies:

        movie_id = movie_enc.inverse_transform([movie_idx])[0]

        print(f"Movie {movie_id} with predicted rating {score:.2f}")



# Example usage

recommend_movies(102)


