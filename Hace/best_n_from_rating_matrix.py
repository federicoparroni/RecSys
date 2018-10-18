import numpy as np

class BestNFromRatingMatrix:

    def __init__(self, d, sp_pred_mat, sp_urm_mat):
        self.arr_tgt_playlists = d.target_playlists_df.values
        self.sp_pred_mat = sp_pred_mat
        self.sp_urm_mat = sp_urm_mat


    def get_best_n_ratings(self, n=10):

        ''' find the best n ratings for a prediction matrix in a numpy matrix

                param_name   | type           | description

        in:     n            | (int)          | number of elements to predict for each playlist
        -----------------------------------------------------
        out:    n_res_matrix | (numpy matrix) | matrix [tgt_playlist.lenght, n+1] for each playlist to predict it gives n tracks
                                                    first column is the id of the playlist the remaining the id of the track

        '''

        n_res_matrix = np.zeros((1, n+1))
        res = np.ndarray(shape=(1, n+1))
        n_res = np.array(res)

        for i in self.arr_tgt_playlists:
            r_urm_mat = self.sp_urm_mat.getrow(i)
            r_pred_mat = self.sp_pred_mat.getrow(i)
            _, c_urm_mat_i = r_urm_mat.nonzero()

            #set to 0 on the prediction matrix the tracks that the playlist just have
            for k in c_urm_mat_i:
                r_pred_mat[0, k] = 0

            for j in range(1, n+1):
                c = r_pred_mat.argmax()
                n_res[0, j] = c
                r_pred_mat[0, c] = 0
            n_res[0, 0] = i
            n_res_matrix = np.concatenate((n_res_matrix, n_res))
        n_res_matrix = n_res_matrix[1:, :]
        return n_res_matrix
