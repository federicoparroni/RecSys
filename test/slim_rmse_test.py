from recommenders.conten_based.SLIM_RMSE import SLIMElasticNetRecommender
import data as d
import inout.importexport as io

urm_train = d.get_urm_train()
target_id = d.get_target_playlists()
urm_test = d.get_urm_test()

recommender = SLIMElasticNetRecommender(urm_train)
recommender.fit()
recommendations = recommender.recommend_batch(userids=target_id)
map10 = recommender.evaluate(recommendations, test_urm=urm_test)
print('map@10: {}'.format(map10))
io.exportcsv(recommendations, 'slim_rmse')
