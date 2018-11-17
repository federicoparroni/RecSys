from scipy.sparse import load_npz
import data as d
import implicit
from recommenders import model_bridge as bridge


"""
Return the MAP10 evaluation for the specified epochs
INPUT
epochs: (array) epochs at which perform the evaluation

OUTPUT
output: array with MAP10 evaluations (epoch, MAP10)
"""
def evaluate_als(epochs, factors, verbose=True):
    # load data
    d = Data()
    #targetUsersIds = d.target_playlists_df['playlist_id'].values
    targetUsersIds = d.target_playlists_df['playlist_id'].values

    # load URM test MAP matrix
    test_urm = load_npz('../raw_data/matrices/sp_urm_test_MAP.npz')

    # get item_user matrix by transposing the URM matrix
    URM = load_npz('../raw_data/matrices/sp_urm_train_MAP.npz')
    item_user_data = URM.transpose()

    evaluations=[]

    # initialize a model
    for e in epochs:
        for f in factors:
            if verbose:
                print('Epoch {} factors {}'.format(e, f))


            model = implicit.als.AlternatingLeastSquares(factors=f, iterations=e)
            model.fit(item_user_data)
            recommendations = bridge.array_of_recommendations(model, target_user_ids=targetUsersIds, urm=URM)

            map10 = evaluate_map(recommendations, test_urm)
            evaluations.append((e, map10))

            if verbose:
                print('Estimated map after {} epochs factors {}: {}'.format(e, f, map10))

    return evaluations


a = d.get_all_playlists()