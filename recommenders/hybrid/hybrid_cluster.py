import clusterize.cluster as cluster
import data.data as data
from recommenders.recommender_base import RecommenderBase
import utils.log as log
import numpy as np

class HybridClusterInteractionsCount(RecommenderBase):

    def __init__(self):
        self.name = 'Hybrid_Cluster'

    def fit(self, clip=7):
        sparse_pl, dense_pl = cluster.cluster_users_by_interactions_count(clip=clip)

        log.success('Cluster 1 (interactions count <= {}): {} playlists'.format(clip, len(sparse_pl)))
        log.success('Cluster 2 (interactions count  > {}): {} playlists'.format(clip, len(dense_pl)))

        # filter target playlists from the 2 clusters
        s1 = set(sparse_pl)
        s2 = set(dense_pl)
        s_target = set(data.get_target_playlists())
        s1_target = s1 & s_target
        s2_target = s2 & s_target
        self.sparse_pl = list(s1_target)
        self.dense_pl = list(s2_target)

    def recommend_batch(self, rec_dense, rec_sparse, verbose=False):

        recommendation_dense = rec_dense.recommend_batch(userids=self.dense_pl)
        recommendation_sparse = rec_sparse.recommend_batch(userids=self.sparse_pl)

        recommendations = np.concatenate((recommendation_dense, recommendation_sparse), axis=0)
        ids = np.concatenate((np.array(self.dense_pl), np.array(self.sparse_pl)))
        t_id = np.reshape(ids, (len(ids), 1))

        return np.concatenate((t_id, recommendations), axis=1)



    def get_r_hat(self):
        pass

    def recommend(self):
        pass

    def run(self):
        pass




