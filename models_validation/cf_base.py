from recommenders.collaborative_filtering.collaborative_filtering_base import CollaborativeFilteringBase
import data
import utils.log as log

urm = data.get_urm_train()
test = data.get_urm_test()
targetids = data.get_target_playlists()

s_plus = CollaborativeFilteringBase.SIM_SPLUS

ks = [100, 200, 300]
alphas = [0.25, 0.5, 0.75]
betas = [0.25, 0.5, 0.75]
ls = [0.25, 0.5, 0.75]
cs = [0.25, 0.5, 0.75]
shrinks = [10, 30, 50, 100]

with open('validation_results/cf_base.txt', 'w') as file:
    for k in ks:
        for a in alphas:
            for b in betas:
                for l in ls:
                    for c in cs:
                        for shrink in shrinks:
                            model = CollaborativeFilteringBase()
                            model.fit(urm, k=k, distance=s_plus, alpha=a, beta=b, c=c, l=l, shrink=shrink)
                            recs = model.recommend_batch(targetids, with_scores=False, verbose=False)

                            map10 = model.evaluate(recs, test_urm=test)
                            
                            logmsg = 'MAP: {} \tknn: {} \ta: {} \tb: {} \tl: {} \tc: {} \tshrink: {}\n'.format(map10,k,a,b,l,c,shrink)
                            log.warning(logmsg)
                            file.write(logmsg)


