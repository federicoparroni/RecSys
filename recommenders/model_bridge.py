from recommenders.progressbar import printProgressBar
import numpy as np

"""
Return the array of recommendations needed by Export function (when model is provided)

@Params
model:            model object
target_user_ids:  array of target user ids
urm:              CSR matrix
@Output
reccomendations:  list of reccomendation for target playlists (user ids)
"""
def array_of_recommendations(model, target_user_ids, urm, include_scores=False, verbose=True):
    # build recommendations array
    recommendations = []
    k=1
    L=len(target_user_ids)
    for userId in target_user_ids:
        rec = model.recommend(userid=userId, user_items=urm, N=10)
        if include_scores:
            recommendations.append([userId, rec])
        else:
            r,scores = zip(*rec)    # zip recommendations and scores
            recommendations.append([userId] + [j for j in r] + [-1 for add_missing in range(10-len(r))])   # create a row: userId | rec1, rec2, rec3, ...

        if verbose:
            printProgressBar(k, L, prefix = 'Building recommendations:', length = 40)
        k+=1

    return recommendations

"""
Return the array of recommendations needed by Export function (when estimated rating matrix is provided)

        param_name   | type           | description

in:     n            | (int)          | number of elements to predict for each playlist
in:     sp_pred_mat      | csr_matrix     | predicted score for tracks in playlists
in:     sp_urm_mat         | csr_matrix     | urm used for training
in:     df_tgt_playlists   | pd's dataframe | target playlists
-----------------------------------------------------
out:    n_res_matrix | (numpy matrix) | matrix [tgt_playlist.lenght, n+1] for each playlist to predict it gives n tracks
                                            first column is the id of the playlist the remaining the id of the track
"""
def get_best_n_ratings(sp_pred_mat, arr_tgt_playlists, sp_urm_mat, n=10, verbose=True):
    sp_pred_mat = sp_pred_mat.tocsr()
    n_res_matrix = np.zeros((1, n+1))
    res = np.ndarray(shape=(1, n+1))
    n_res = np.array(res)
    tp = arr_tgt_playlists['playlist_id'].values

    L=len(tp)
    k=1
    for i in tp:
        r_urm_mat = sp_urm_mat.getrow(i)
        r_pred_mat = sp_pred_mat.getrow(i).todense()
        c_urm_mat_i = r_urm_mat.indices

        # set to 0 on the prediction matrix the tracks that the playlist just have
        for k in c_urm_mat_i:
            r_pred_mat[0, k] = 0

        for j in range(1, n+1):
            c = r_pred_mat.argmax()
            n_res[0, j] = c
            r_pred_mat[0, c] = 0
        n_res[0, 0] = i
        n_res_matrix = np.concatenate((n_res_matrix, n_res))

        if verbose:
            printProgressBar(k, L, prefix = 'Building recommendations:', length=24)
        k+=1
    n_res_matrix = n_res_matrix[1:, :]
    return n_res_matrix.tolist()
