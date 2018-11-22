from recommenders.collaborative_filtering.collaborative_filtering import CollaborativeFiltering
import data.data as data
import utils.log as log

s_plus = CollaborativeFiltering.SIM_SPLUS

ks = [100, 200, 300]
alphas = [0.25, 0.5, 0.75]
betas = [0.25, 0.5, 0.75]
ls = [0.25, 0.5, 0.75]
cs = [0.25, 0.5, 0.75]
shrinks = [0, 10, 30]

i=0
tot=len(ks)*len(alphas)*len(betas)*len(ls)*len(cs)*len(shrinks)

with open('splus_validation.txt', 'w') as file:
    for k in ks:
        for a in alphas:
            for b in betas:
                for l in ls:
                    for c in cs:
                        for shrink in shrinks:
                            model = CollaborativeFiltering()
                            recs, map10 = model.run(distance=s_plus, k=k, shrink=shrink, alpha=a, beta=b, c=c, l=l, verbose=False)
                            logmsg = 'MAP: {} \tknn: {} \ta: {} \tb: {} \tshrink: {}\n'.format(map10,k,a,b,shrink)
                            log.warning(logmsg)
                            file.write(logmsg)
                            
                            i+=1
                            print('{0:.2f} completed: {}/{}'.format(i/tot*100,i,tot))