from recommenders.collaborative_filtering.alternating_least_square import AlternatingLeastSquare
import data as d
from inout import importexport
from evaluation.evaluate_als import evaluate_als


"""
recommender = AlternatingLeastSquare(urm_train)
recommender.fit(iterations=1)
recommendations = recommender.recommend_batch(userids=target_id)
map10 = recommender.evaluate(recommendations, test_urm=urm_test)
print(map10)
importexport.exportcsv(recommendations, path='submissions', name='')
"""

f = [75, 100, 200, 400]
r = [0.01, 0.05, 0.1, 0.2, 0.5]
i = [50, 100, 200, 300]
a = [5, 10, 20, 40]
target_id = d.get_all_playlists()
urm_train = d.get_urm_train()
urm_test = d.get_urm_test()

evaluate_als(f, r, i, a, target_id, urm_train, urm_test)
