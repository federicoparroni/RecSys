import numpy as np

class BestNFromRatingMatrix:

    def __init__(self, d, sp_pred_mat):
        self.arr_tgt_playlists = d.target_playlists_df.values
        self.sp_pred_mat = sp_pred_mat

    '''
        find the best n ratings for a prediction matrix in a numpy matrix
    '''
    def get_best_n_ratings(self, n):
        n_res_matrix = np.zeros((1, 11))
        res = np.ndarray(shape=(1, 11))
        n_res = np.array(res)

        for i in self.arr_tgt_playlists:
            r = self.sp_pred_mat.getrow(i)
            for j in range(1, 11):
                c = r.argmax()
                n_res[0, j] = c
                r[0, c] = 0
            n_res[0, 0] = i
            n_res_matrix = np.concatenate((n_res_matrix, n_res))
        n_res_matrix = n_res_matrix[1:, :]
        return n_res_matrix
