from recommenders.collaborative_filtering.alternating_least_square import AlternatingLeastSquare
import data as d
import time
import os

def validate_als(factors_array, regularization_array, iterations_array, alpha_val_array, userids,
                 urm_train, urm_test, filter_already_liked=True, items_to_exclude=[], N=10,
                 verbose=True, write_on_file=True):
    """

    :param factors_array
    :param regularization_array
    :param iterations_array
    :param alpha_val_array
    :param userids: id of the users to take into account during evaluation
    :param urm_train: matrix on which train the model
    :param urm_test: matrix in which test the model
    :param filter_already_liked:
    :param items_to_exclude:
    :param N: evaluate on map@10
    :param verbose:
    :param write_on_file:
    ----------- 
    :return: _
    """


    #create the initial model
    recommender = AlternatingLeastSquare(urm_train)

    path = 'validation_results/'
    name = 'als'
    folder = time.strftime('%d-%m-%Y')
    filename = '{}/{}/{}{}.csv'.format(path, folder, name, time.strftime('_%H-%M-%S'))
    # create dir if not exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as out:
        for f in factors_array:
            for r in regularization_array:
                for i in iterations_array:
                    for a in alpha_val_array:

                        #train the model with the parameters
                        if verbose:
                            print('\n\nTraining ALS with\n Factors: {}\n Regulatization: {}\n'
                                  'Iterations: {}\n Alpha_val: {}'.format(f, r, i, a))
                            print('\n training phase...')
                        recommender.fit(f, r, i, a)

                        #get the recommendations from the trained model
                        recommendations = recommender.recommend_batch(userids=userids, N=N, filter_already_liked=filter_already_liked,
                                                                      items_to_exclude=items_to_exclude)
                        #evaluate the model with map10
                        map10 = recommender.evaluate(recommendations, test_urm=urm_test)
                        if verbose:
                            print('map@10: {}'.format(map10))

                        #write on external files on folder models_validation
                        if write_on_file:
                            out.write('\n\nFactors: {}\n Regulatization: {}\n Iterations: {}\n '
                                      'Alpha_val: {}\n evaluation map@10: {}'.format(f, r, i, a, map10))





