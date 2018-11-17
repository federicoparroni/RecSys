# from surprise import SVDpp
# from recommenders.recommender_base import RecommenderBase
#
#
# class SvdPP(RecommenderBase):
#
#     def __init__(self, URM):
#         self.urm = URM
#         self.algo = SVDpp()
#
# from surprise import SVD
# from surprise import Dataset
# from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed),
# We'll use the famous SVD algorithm.

from collections import defaultdict
from surprise import SVD
import numpy as np
import data as d
from surprise import Reader
import pandas as pd


def get_top_n(predictions, n=10):
    ''' Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First train an SVD algorithm on the movielens dataset.
# data = Dataset.load_builtin('ml-100k')

r, c = d.get_urm().nonzero()
ones = np.ones(len(r), dtype=np.int32)
d = np.vstack((r, c, ones)).transpose()
df = pd.DataFrame(d)
reader = Reader(rating_scale=(0, 1))


trainset = df.build_full_trainset()
algo = SVD(n_factors = 20, verbose=True, n_epochs = 1)
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

print("get predictions")

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])