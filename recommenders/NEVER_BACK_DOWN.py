from scipy.sparse import load_npz
from data import Data
import numpy as np
import implicit # The Cython library
from io.export_rec import Export


""" Implementation of Alternating Least Squares with implicit data. We iteratively
    compute the user (x_u) and item (y_i) vectors using the following formulas:

    x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
    y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

    Args:
        sparse_data (csr_matrix): Our sparse user-by-item matrix

        alpha_val (int): The rate in which we'll increase our confidence
        in a preference with more interactions.

        iterations (int): How many times we alternate between fixing and
        updating our user and item vectors

        lambda_val (float): Regularization value

        features (int): How many latent features we want to compute.

    Returns:
        X (csr_matrix): user vectors of size users-by-features

        Y (csr_matrix): item vectors of size items-by-features
"""


# The implicit library expects data as a item-user matrix so we
# create two matricies, one for fitting the model (item-user)
# and one for recommendations (user-item)
URM = load_npz('../raw_data/matrices/urm.npz')
URM_T = URM.T

TEST_URM = load_npz('../raw_data/matrices/urm_test.npz')

sparse_item_user = URM_T
sparse_user_item = URM

# Initialize the als model and fit it using the sparse item-user matrix
model = implicit.als.AlternatingLeastSquares(factors=400, regularization=0.01, iterations=50)

# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 30
data_conf = (sparse_item_user * alpha_val).astype('double')

# Fit the model
model.fit(data_conf)

# Get the user and item vectors from our trained model
user_vecs = model.user_factors
item_vecs = model.item_factors

d = Data()
target_id = d.target_playlists_df['playlist_id'].values
#target_id = d.all_playlists['playlist_id'].values



def recommend(URM, user_vecs, item_vecs, target_id, n=10, exclude_seen=True):
    # compute the scores using the dot product
    user_profile_batch = URM[target_id]

    scores_array = np.dot(user_vecs[target_id], item_vecs.T)

    # To exclude seen items perform a boolean indexing and replace their score with -inf
    # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
    # recommended
    if exclude_seen:
        scores_array[user_profile_batch.nonzero()] = -np.inf

    ranking = np.zeros((scores_array.shape[0], n), dtype=np.int)

    for row_index in range(scores_array.shape[0]):
        scores = scores_array[row_index]

        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]

    return ranking


#test
recommendations_no_id = recommend(URM, user_vecs, item_vecs, target_id)
np_target_id = np.array(target_id)
target_id_t = np.reshape(np_target_id, (len(np_target_id),1))
recommendations = np.concatenate((target_id_t, recommendations_no_id), axis=1)


#map10_evaluation = evaluate_map(recommendations, TEST_URM)
#print("HOPE -> {}".format(map10_evaluation))

# export
Export.export(np.array(recommendations), path='../submissions/', name='als')
print('file exported')