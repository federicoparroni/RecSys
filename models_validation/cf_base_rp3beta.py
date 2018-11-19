from recommenders.collaborative_filtering.collaborative_filtering_base import CollaborativeFilteringBase
import data
import utils.log as log

urm = data.get_urm_train()
test = data.get_urm_test()
targetids = data.get_target_playlists()

rp3beta = CollaborativeFilteringBase.SIM_RP3BETA

ks = [100, 200, 300]
alphas = [0.25, 0.5, 0.75]
betas = [0.25, 0.5, 0.75]
shrinks = [50, 80, 100, 150]

with open('validation_results/cf_base_rp3beta.txt', 'w') as file:
    for k in ks:
        for a in alphas:
            for b in betas:
                for shrink in shrinks:
                    model = CollaborativeFilteringBase()
                    model.fit(urm, k=k, distance=rp3beta, alpha=a, beta=b, shrink=shrink)
                    recs = model.recommend_batch(targetids, verbose=False)

                    map10 = model.evaluate(recs, test_urm=test)
                    
                    logmsg = 'MAP: {} \tknn: {} \ta: {} \tb: {} \tshrink: {}\n'.format(map10,k,a,b,shrink)
                    log.warning(logmsg)
                    file.write(logmsg)
