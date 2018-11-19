#knn: 100 	a: 0.25 	b: 0.25 	l: 0.5 	c: 0.5 	shrink: 10

from recommenders.collaborative_filtering.collaborative_filtering_base import CollaborativeFilteringBase
import data
import utils.log as log
import inout.importexport as export

urm = data.get_urm_train()
test = data.get_urm_test()
targetids = data.get_target_playlists()
#targetids = data.get_all_playlists()

model = CollaborativeFilteringBase()
model.fit(urm, k=100, distance=CollaborativeFilteringBase.SIM_SPLUS, alpha=0.25, beta=0.25, c=0.5, l=0.5, shrink=10)
recs = model.recommend_batch(targetids, verbose=False)

map10 = model.evaluate(recs, test_urm=test)

logmsg = 'MAP: {}'.format(map10)
log.warning(logmsg)

#export.exportcsv(recs, 'submission', 'splus_np_target')
