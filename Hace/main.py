from data import Data
from matrix import M
from algorithms.cos_similarity import CosineSimilarityCB
from scipy.sparse import load_npz


from best_n_from_rating_matrix import BestNFromRatingMatrix
from helpers.export import Export

# ===============================================================

d = Data()
m = M()

sp_urm = load_npz('../dataset/saved_matrices/sp_urm.npz')
sp_icm = load_npz('../dataset/saved_matrices/sp_icm.npz')
print('loaded matrices')

m.create_Sknn(sp_icm)

sp_pred_mat = CosineSimilarityCB.predict(sp_icm, sp_urm, knn=50)
print('computed prediction matrices')

bestn = BestNFromRatingMatrix(d, sp_pred_mat, sp_urm)

n_res_matrix = bestn.get_best_n_ratings(10)

n_res_matrix = bestn.get_best_n_ratings()
print('res_matrix create')

print('starting export')

Export.export(n_res_matrix, path='submissions/')
