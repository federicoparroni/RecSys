from data import Data
from matrix import M
from algorithms.cos_similarity import CosineSimilarityCB
from algorithms.collaborative_filtering import CosineSimilarityCF
from scipy.sparse import load_npz


from best_n_from_rating_matrix import BestNFromRatingMatrix
from helpers.export import Export

# ===============================================================

d = Data()
m = M()

sp_urm = load_npz('../dataset/saved_matrices/sp_urm.npz')
sp_icm = load_npz('../dataset/saved_matrices/sp_icm.npz')
print('loaded matrices')


#sp_pred_mat1 = CosineSimilarityCB.predict(sp_icm, sp_urm, shrink_term=30)

#todo
sp_pred_mat2 = CosineSimilarityCF.predict(sp_urm, shrink_term=30)

print('computed prediction matrices')




#sp_pred_mat = sp_pred_mat1 + sp_pred_mat2

bestn = BestNFromRatingMatrix(d, sp_pred_mat2, sp_urm)

n_res_matrix = bestn.get_best_n_ratings(10)

print('res_matrix create')

print('starting export')

Export.export(n_res_matrix, path='submissions/')
