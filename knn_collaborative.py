from scipy.sparse import load_npz
#from spotify_recsys_challenge.utils.datareader import Datareader
#from spotify_recsys_challenge.utils.evaluator import  Evaluator
#from spotify_recsys_challenge.utils.submitter import Submitter
#from spotify_recsys_challenge.utils.post_processing import  eurm_to_recommendation_list_submission
#from spotify_recsys_challenge.utils.post_processing import  eurm_to_recommendation_list
from spotify_recsys_challenge.recommenders.knn_collaborative_user import Knn_collabrative_user
import spotify_recsys_challenge.recommenders.similarity.similarity as sm
import scipy.sparse as sps
import sys
import numpy as np
import pandas as pd
from helpers.model_bridge import get_best_n_ratings
from evaluation.map_evaluation import evaluate_map
import spotify_recsys_challenge.utils.sparse as ut
from spotify_recsys_challenge.utils.evaluator import  Evaluator


'''
This file contains just an example on how to run the algorithm.
The parameter used are just the result of a first research of the optimum value.
To run this file just set the parameter at the start of the main function or set from console as argv parameter.
As argv you can even set mode of execution (online, offline) and the name of the result file
'''

if __name__ == '__main__':


    ### Select execution mode: 'offline', 'online' ###
    mode = "offline"
    name = "knn_collaborative_user"
    knn = 250
    topk = 800

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        name = sys.argv[2]
        knn = int(sys.argv[3])
        topk = int(sys.argv[4])

    complete_name = mode+"_"+name+"_knn="+str(knn)+"_topk="+str(topk)

    #Recommender algorithm initialization
    rec = Knn_collabrative_user()

    #Getting for the recommender algorithm
    urm = load_npz('dataset/saved_matrices/sp_urm_train_MAP.npz')
    urm.data = np.ones(len(urm.data))
    tp = pd.read_csv('dataset/target_playlists.csv')
    pid = tp['playlist_id'].values

    # Depopularize
    top = urm.sum(axis=0).A1
    mask = np.argsort(top)[::-1][:2000]
    ut.inplace_set_cols_zero(urm, mask)

    #Fitting data
    rec.fit(urm, pid)

    # load validation
    test_urm = load_npz('dataset/saved_matrices/sp_urm_test_MAP.npz')

    for sim_metric in ['jaccard', 'dice', 'tversky', 'as_cosine']:
        #Computing similarity/model
        rec.compute_model(top_k=knn, sm_type=sim_metric, shrink=10, alpha=0.1, beta=1.5, binary=True, verbose=True)

        #Computing ratings
        rec.compute_rating(top_k=topk, verbose=True, small=False)

        #evaluation and saving
        recommendations = get_best_n_ratings(rec.eurm, arr_tgt_playlists=tp, sp_urm_mat=urm)

        # VALIDATION: load URM test MAP matrix
        map10 = evaluate_map(recommendations, test_urm)
        print('Estimated map --> {}'.format(map10))


