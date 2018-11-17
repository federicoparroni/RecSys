from scipy.sparse import load_npz
from recommenders.model_bridge import get_best_n_ratings
from recommenders.cosine_similarity_CB import CosineSimilarityCB
from evaluation.map_evaluation import evaluate_map
from inout.export_rec import Export
import pandas as pd

def evaluate_CB(shrink = [0, 1, 2, 3, 4, 5], knn = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):

    sp_train_urm = load_npz('../raw_data/matrices/urm_train.npz')
    sp_icm = load_npz('../raw_data/matrices/icm.npz')
    sp_test_urm = load_npz('../raw_data/matrices/urm_test.npz')

    for s in shrink:
        for k in knn:
            sp_pred_mat1 = CosineSimilarityCB.predict(sp_icm, sp_train_urm, knn=k, shrink_term=s)
            p = pd.read_csv('../raw_data/all_playlist.csv')
            bestn = get_best_n_ratings(sp_pred_mat1, p, sp_train_urm)
            map = evaluate_map(bestn, sp_test_urm)
            print('map with shrink={}, knn={}: {}'.format(s, k, map))

    Export.export(bestn, path='../submissions/', name='CB_0034')

evaluate_CB(shrink = [56], knn=[110])
