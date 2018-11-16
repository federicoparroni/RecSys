import numpy as np
from scipy.sparse import csr_matrix

class ElectionMethods:

    @staticmethod
    def borda_count(recommendations_array, weights_array, N=10):
        ''' given an array of result matrices, compute only one result matrix using the borda count voting methodology
                        param_name          | type                   | description

                :param recommendations_array  | array of matrices      | array of result matrices from various methodology
                        n                   | int                    | number of tracks to which gives points (length of recommendation list)

                ----------------------------------------------------------
                out:    res_matrix          | (numpy matrix)         | matrix [tgt_playlist.lenght, n+1] for each playlist to predict it
                                                                        gives n tracks first column is the id of the playlist
                                                                        the remaining the id of the tracks
        '''
        n_res_matrix = np.zeros((1, N + 1))

        res = np.ndarray(shape=(1, N+1))
        n_res = np.array(res)

        for i in range(len(recommendations_array[0])): #cycle between rows
            N_TRACKS = 20634 + 1
            temp = np.zeros(N_TRACKS)
            for j in range(len(recommendations_array)): #cycle between matrices
                weight = weights_array[j]
                h = N
                for k in recommendations_array[j][i, 1:]: #cycle between columns
                    temp[k] += h * weight
                    h -= 1
            sp_temp = csr_matrix(temp)

            n_res[0, 0] = recommendations_array[0][i, 0] # set playlist_id

            for l in range(N):
                c = sp_temp.argmax()
                n_res[0, l+1] = c
                sp_temp[0, c] = 0
            n_res_matrix = np.concatenate((n_res_matrix, n_res))

        n_res_matrix = n_res_matrix[1:, :]

        return n_res_matrix


    @staticmethod
    def borda_count_on_scores(recommendations_array, weights_array, N=10):
        ''' given an array of result matrices, compute only one result matrix using the borda count voting methodology
                        param_name          | type                   | description

                in:     res_matrices_array  | array of matrices      | array of result matrices from various methodology
                        n                   | int                    | number of tracks to which gives points (length of recommendation list)

                ----------------------------------------------------------
                out:    res_matrix          | (numpy matrix)         | matrix [tgt_playlist.lenght, n+1] for each playlist to predict it
                                                                        gives n tracks first column is the id of the playlist
                                                                        the remaining the id of the tracks
        '''
        n_res_matrix = []

        for i in range(len(recommendations_array[0])): #cycle between rows
            n_res = [recommendations_array[0][i][0]] # set playlist_id
            N_TRACKS = 20634 + 1
            temp = np.zeros(N_TRACKS)
            for j in range(len(recommendations_array)): #cycle between matrices
                weight = weights_array[j]
                ids_scores = recommendations_array[j][i][1]
                h = N
                for id_score_pair in ids_scores: #cycle between columns
                    track_id = id_score_pair[0]
                    score = id_score_pair[1]
                    temp[track_id] += h * weight * score
                    h -= 1
            sp_temp = csr_matrix(temp)

            for l in range(N):
                c = sp_temp.argmax()
                n_res.append(c)
                sp_temp[0, c] = 0
            
            n_res_matrix.append(n_res)

        return n_res_matrix
