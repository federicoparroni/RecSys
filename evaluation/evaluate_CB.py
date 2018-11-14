from data import Data
from scipy.sparse import load_npz
from helpers.model_bridge import get_best_n_ratings
from algorithms.cosine_similarity_CB import CosineSimilarityCB
from evaluation.map_evaluation import evaluate_map

def evaluate_CB(shrink = [0, 1, 2, 3, 4, 5], knn = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    d = Data()

    sp_train_urm = load_npz('../dataset/saved_matrices/sp_urm_train_MAP.npz')
    sp_icm = load_npz('../dataset/saved_matrices/sp_icm.npz')
    sp_test_urm = load_npz('../dataset/saved_matrices/sp_urm_test_MAP.npz')

    for s in shrink:
        for k in knn:
            sp_pred_mat1 = CosineSimilarityCB.predict(sp_icm, sp_train_urm, knn=10, shrink_term=30)

            bestn = get_best_n_ratings(sp_pred_mat1, d.target_playlists_df, sp_train_urm)

            map = evaluate_map(bestn, sp_test_urm)

            print('map with shrink={}, knn={}: {}'.format(s, k, map))

evaluate_CB()