from recommenders.collaborative_filtering.alternating_least_square import AlternatingLeastSquare
import data.data as d
from inout import importexport



target_id = d.get_all_playlists()
urm_train = d.get_urm_train()
urm_test = d.get_urm_test()


recommender = AlternatingLeastSquare(urm_train)
recommender.fit(iterations=1)
recommender.save_r_hat()
recommendations = recommender.recommend_batch(userids=target_id)
map10 = recommender.evaluate(recommendations, test_urm=urm_test)
print(map10)
importexport.exportcsv(recommendations, path='submissions', name='')


