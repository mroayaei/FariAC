# Dependencies

import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

import time

from envs import OfflineEnv
from recommender import DRRAgent
from memory_profiler import profile

import os

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-100k')
STATE_SIZE = 10
MAX_EPISODE_NUM = 100
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def example():
    print('Data loading...')
    # Loading datasets
    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in
                   open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]
    ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=np.uint32)
    movies_df = pd.DataFrame(movies_list, columns=['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")
    # شناسه فیلم به عنوان عنوان فیلم
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    ratings_df = ratings_df.applymap(int)

    # به ترتیب فیلم های تماشا شده توسط کاربران مرتب شده است
    users_dict = np.load('data/user_dict.npy', allow_pickle=True)

    # طول تاریخچه فیلم برای هر کاربر
    users_history_lens = np.load('data/users_histroy_len.npy')

    users_num = max(ratings_df["UserID"]) + 1
    items_num = max(ratings_df["MovieID"]) + 1

    print("Run Time: " + str(time.time()))
    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k: users_dict.item().get(k) for k in range(1, train_users_num + 1)}
    train_users_history_lens = users_history_lens[:train_users_num]

    print('DONE!')
    time.sleep(2)
    env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=False)

if __name__ == "__main__":
    example()