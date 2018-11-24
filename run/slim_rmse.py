from recommenders.collaborative_filtering.SLIM_RMSE import SLIMElasticNetRecommender
import data.data as d
import inout.importexport as io

urm = d.get_urm()
urm_train = d.get_urm_train()
target_id = d.get_all_playlists()
urm_test = d.get_urm_test()
t_id = d.get_target_playlists()


recommender = SLIMElasticNetRecommender(urm)
recommender.fit(topK=100, alpha=1e-4, l1_ratio=0.1, max_iter=100, tol=1e-4)
recommender.save_r_hat()
#recommendations = recommender.recommend_batch(userids=t_id)
#map10 = recommender.evaluate(recommendations, test_urm=urm_test)
#print('map@10: {}'.format(map10))
#io.exportcsv(recommendations, path='submissions', name='slim_rmse')


