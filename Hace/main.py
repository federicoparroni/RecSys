from data import Data
from matrix import M
from algorithms.cos_similarity import CosineSimilarity
from scipy.sparse import load_npz

from Hace import best_n_from_rating_matrix
from helpers.export import Export

# ===============================================================

d = Data()
m = M()

# sp_urm = load_npz('../dataset/saved_matrices/sp_urm.npz')
sp_icm = load_npz('../dataset/saved_matrices/sp_icm.npz')

m.create_Sknn(sp_icm)

#sp_pred_mat = CosineSimilarity.predict(sp_icm, sp_urm, knn=50)

#b = best_n_from_rating_matrix(d, sp_pred_mat)
#n_res_matrix = b.get_best_n_ratings(10)

#Export.export(n_res_matrix, path='submissions/')
