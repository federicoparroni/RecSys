from recommenders.collaborative_filtering.pure_SVD import Pure_SVD
import data as d
from inout.importexport import exportcsv

urm = d.get_urm()
urm_train = d.get_urm_train()
target_id = d.get_all_playlists()
urm_test = d.get_urm_test()
t_id = d.get_target_playlists()


recommender = Pure_SVD(urm_train)
recommender.fit(num_factors=250)
recommendations = recommender.recommend_batch(userids=target_id)
map10 = recommender.evaluate(recommendations, test_urm=urm_test)
print('map@10: {}'.format(map10))
exportcsv(recommendations, path='submissions', name='slim_rmse')
