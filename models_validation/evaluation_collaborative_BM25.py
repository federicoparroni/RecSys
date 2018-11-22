from scipy.sparse import load_npz
from data import data.data as data
import implicit
from recommenders import model_bridge as M
from evaluation.map_evaluation import evaluate_map
import numpy as np

K = [30, 70, 150, 250, 300, 400]
K1 = np.arange(0.1, 1.1, 0.1)
B = np.arange(0.1, 0.3, 0.1)

def evaluate_collaborative_BM25(K, K1, B, verbose=True):
    #creating the file where to write
    out = open('evaluation_collaborative_BM25', 'w')

    # load data
    d = Data()
    targetUsersIds = d.target_playlists_df['playlist_id'].values
    all_id = d.all_playlists['playlist_id'].values

    # get item_user matrix by transposing the URM matrix
    URM = load_npz('../raw_data/matrices/sp_urm_train_MAP.npz')
    item_user_data = URM.transpose()

    # load URM test MAP matrix
    test_urm = load_npz('../raw_data/matrices/sp_urm_test_MAP.npz')
    print('> data loaded')

    for k in K:
        for k1 in K1:
            for b in B:

                # initialize a model (BM25 metric)
                model = implicit.nearest_neighbours.BM25Recommender(k, k1, b) #DEFAULT K=50, K1=1.2, B=.75

                # train the model on a sparse matrix of item/user/confidence weights
                model.fit(item_user_data)

                # build recommendations array
                recommendations = M.array_of_recommendations(model, target_user_ids=all_id, urm=URM)

                map10 = evaluate_map(recommendations, test_urm)
                if(map10 > 0.0011):
                    print('Estimated map K {} K1 {} B {} --> {}'.format(k, k1, b, map10))

                #if verbose:
                 #   print('Estimated map K {} K1 {} B {} --> {}'.format(k, k1, b, map10))
                #print('Estimated map K {} K1 {} B {} --> {}'.format(k, k1, b, map10), file=out)


def evaluate_collaborative_cosine_recommender(verbose=True):
    #creating the file where to write
    #out = open('evaluation_collaborative_cosine', 'w')

    # load data
    d = Data()
    targetUsersIds = d.target_playlists_df['playlist_id'].values
    all_playlists = d.all_playlists['playlist_id'].values

    # get item_user matrix by transposing the URM matrix
    URM = load_npz('../raw_data/matrices/sp_urm_train_MAP.npz')
    item_user_data = URM.transpose()

    # load URM test MAP matrix
    test_urm = load_npz('../raw_data/matrices/sp_urm_test_MAP.npz')
    print('> data loaded')


    # initialize a model (BM25 metric)
    model = implicit.nearest_neighbours.TFIDFRecommender()

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(item_user_data)

    # build recommendations array
    recommendations = M.array_of_recommendations(model, target_user_ids=all_playlists, urm=URM)

    map10 = evaluate_map(recommendations, test_urm)

    if verbose:
        print('Estimated map --> {}'.format(map10))



#evaluate_collaborative_BM25(K, K1, B)
evaluate_collaborative_cosine_recommender()