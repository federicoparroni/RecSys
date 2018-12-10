from MF_BPR import fit_MFBPR
import data.data as d
r = fit_MFBPR()
r.fit(d.get_urm_train(),
      epochs=50,
      n_factors=150,
      learning_rate=1e-1, 
      user_regularization=1e-4,
      positive_item_regularization=1e-4,
      negative_item_regularization=1e-4)