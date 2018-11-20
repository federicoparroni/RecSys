from recommenders.distance_based_recommender import DistanceBasedRecommender
import run.cf
import data
import utils.log as log


s_plus = DistanceBasedRecommender.SIM_SPLUS

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
                            recs, map10 = run.cf.run(distance=s_plus, k=k, shrink=shrink, alpha=a, beta=b, c=c, l=l, verbose=False)
                            logmsg = 'MAP: {} \tknn: {} \ta: {} \tb: {} \tshrink: {}\n'.format(map10,k,a,b,shrink)
                            log.warning(logmsg)
                            file.write(logmsg)
