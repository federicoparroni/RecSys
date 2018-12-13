from MF_BPR import fit_MFBPR
import data.data as d
r = fit_MFBPR()
r.fit(d.get_urm_train_1(),
      epochs=50,
      n_factors=150,
      learning_rate=1e-3, 
      user_regularization=1e-3,
      positive_item_regularization=1e-3,
      negative_item_regularization=1e-3)