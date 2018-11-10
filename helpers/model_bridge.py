from helpers.progressbar import printProgressBar

"""
Return the array of recommendations needed by Export function

@Params
model:            model object
target_user_ids:  array of target user ids
urm:              CSR matrix
"""
def array_of_recommendations(model, target_user_ids, urm, verbose=True):
  # build recommendations array
  recommendations = []
  k=1
  L=len(target_user_ids)
  for userId in target_user_ids:
    rec = model.recommend(userid=userId, user_items=urm, N=10)
    r,scores = zip(*rec)    # zip recommendations and scores
    recommendations.append([userId] + [j for j in r])   # create a row: userId | rec1, rec2, rec3, ...

    if verbose:
      printProgressBar(k, L, prefix = 'Building recommendations:', length = 40)
    k+=1
  
  return recommendations