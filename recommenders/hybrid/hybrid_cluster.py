import clusterize.cluster as cluster
import data
import run.cb as cb
import run.cf as cf
from recommenders.recommender_base import RecommenderBase
import inout.importexport as export
import utils.log as log

CLIP = 7

# get the 2 clusterized datasets
urm = data.get_urm_train()
target_playlists = data.get_target_playlists()

p1, p2 = cluster.cluster_users_by_interactions_count(clip=CLIP)
urm1 = urm[p1]
urm2 = urm[p2]

# filter target playlists from the 2 clusters
s1 = set(p1)
s2 = set(p2)
s_target = set(target_playlists)
s1_target = s1 & s_target
s2_target = s2 & s_target
p1 = list(s1_target)
p2 = list(s2_target)

log.success('Cluster 1 (interactions count <= {}): {} playlists'.format(CLIP, len(p1)))
log.success('Cluster 2 (interactions count  > {}): {} playlists'.format(CLIP, len(p2)))

""" run the models on the 2 different clusters """
#recs1, map1 = cb.run(distance='splus', targetids=p1, k=100, alpha=0.25, beta=0.25, l=0.5, c=0.5, shrink=10, export=False)
#recs2, map2 = cf.run(distance='splus', targetids=p2, k=100, alpha=0.25, beta=0.25, l=0.5, c=0.5, shrink=10, export=False)
recs1, map1 = cb.run(distance='splus', targetids=p1, k=100, alpha=0.25, beta=0.25, l=0.5, c=0.5, shrink=10, export=False)
recs2, map2 = cf.run(distance='splus', urm_train=urm2, targetids=p2, k=100, alpha=0.25, beta=0.25, l=0.5, c=0.5, shrink=10, export=False)

recs = recs1 + recs2

test_urm = data.get_urm_test()
map10 = RecommenderBase.evaluate(self=None, recommendations=recs, test_urm=test_urm)

export.exportcsv(recs, path='submission', name='hybrid_cluster')
