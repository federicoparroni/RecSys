from recommenders.collaborative_filtering.collaborative_filtering_base import CollaborativeFilteringBase
import data
import utils.log as log

target_ids = data.get_all_playlists()
urm_train = data.get_urm_train()
urm_test = data.get_urm_test()

sim = CollaborativeFilteringBase.SIM_TVERSKY

ks = [100, 200, 300]
alphas = [0.25, 0.5, 0.75]
betas = [0.25, 0.5, 0.75]
shrinks = [50, 80, 100, 150]

#with open('validation_results/cf_base_twersky.txt', 'w') as file:
for k in ks:
    for a in alphas:
        for b in betas:
            for shrink in shrinks:
                model = CollaborativeFilteringBase()
                model.fit(urm_train, k=k, distance=sim, alpha=a, beta=b)
                recs = model.recommend_batch(target_ids, verbose=False)

                map10 = model.evaluate(recs, test_urm=urm_test)
                
                logmsg = 'MAP: {} \tknn: {} \ta: {} \tb: {} \tshrink: {}\n'.format(map10,k,a,b,shrink)
                log.warning(logmsg)
                    #file.write(logmsg)
