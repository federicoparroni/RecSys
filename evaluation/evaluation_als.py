from scipy.sparse import load_npz
from data import Data
import implicit
from helpers import model_bridge as M
from evaluation.map_evaluation import evaluate_map

"""
Return the MAP10 evaluation for the specified epochs
INPUT
epochs: (array) epochs at which perform the evaluation

OUTPUT
output: array with MAP10 evaluations (epoch, MAP10)
"""
def evaluate_als(epochs=[5, 10, 15, 20, 25, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 500], verbose=True):
    # load data
    d = Data()
    targetUsersIds = d.target_playlists_df['playlist_id'].values

    # load URM test MAP matrix
    test_urm = load_npz('../dataset/saved_matrices/sp_urm_test_MAP.npz')

    # get item_user matrix by transposing the URM matrix
    URM = load_npz('../dataset/saved_matrices/sp_urm.npz')
    item_user_data = URM.transpose()

    evaluations=[]

    # initialize a model
    for e in epochs:
        if verbose:
            print('Epoch {}'.format(e))

        model = implicit.als.AlternatingLeastSquares(factors=150, iterations=e)
        model.fit(item_user_data)
        recommendations = M.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)
    
        map10 = evaluate_map(recommendations, test_urm)
        evaluations.append((e, map10))

        if verbose:
            print('Estimated map after {} epochs: {}'.format(e, map10))
    
    return evaluations


evaluate_als()