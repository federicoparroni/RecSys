from scipy.sparse import load_npz
from data import Data
import implicit
from helpers import model_bridge as M
from evaluation.map_evaluation import evaluate_map

def evaluate_collaborative_BM25(K=[25, 50, 100, 200], K1=[0.8, 1.2, 1.6, 2], B=[0.3, 0.5, 0.75, 0.9], verbose=True):
    #creating the file where to write
    out = open('evaluation_collaborative_BM25', 'w')

    # load data
    d = Data()
    targetUsersIds = d.target_playlists_df['playlist_id'].values

    # get item_user matrix by transposing the URM matrix
    URM = load_npz('../dataset/saved_matrices/sp_urm.npz')
    item_user_data = URM.transpose()

    # load URM test MAP matrix
    test_urm = load_npz('../dataset/saved_matrices/sp_urm_test_MAP.npz')
    print('> data loaded')

    for k in K:
        for k1 in K1:
            for b in B:

                # initialize a model (BM25 metric)
                model = implicit.nearest_neighbours.BM25Recommender(k, k1, b) #DEFAULT K=50, K1=1.2, B=.75

                # train the model on a sparse matrix of item/user/confidence weights
                model.fit(item_user_data)

                # build recommendations array
                recommendations = M.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)

                map10 = evaluate_map(recommendations, test_urm)

                if verbose:
                    print('Estimated map K {} K1 {} B {} --> {}'.format(k, k1, b, map10))
                print('Estimated map K {} K1 {} B {} --> {}'.format(k, k1, b, map10), file=out)



evaluate_collaborative_BM25()