import clusterize.cluster as cluster
import utils.log as log
import data.data as data
import inout.importexport as imp
import pandas as pd
import time as t

def cluster_ensemble(clip, path_sparse, path_dense):
    sparse_pl, dense_pl = cluster.cluster_users_by_interactions_count(clip=clip)

    log.success('Cluster 1 (interactions count <= {}): {} playlists'.format(clip, len(sparse_pl)))
    log.success('Cluster 2 (interactions count  > {}): {} playlists'.format(clip, len(dense_pl)))

    # filter target playlists from the 2 clusters
    s1 = set(sparse_pl)
    s2 = set(dense_pl)
    s_target = set(data.get_target_playlists())
    s1_target = s1 & s_target
    s2_target = s2 & s_target

    sparse_pl = pd.DataFrame({'playlist_id':list(s1_target)})
    dense_pl= pd.DataFrame({'playlist_id': list(s2_target)})




    df_sparse = pd.read_csv(path_sparse)
    df_dense = pd.read_csv(path_dense)

    cluster1 = df_sparse.merge(sparse_pl)
    cluster2 = df_dense.merge(dense_pl)

    final = pd.concat([cluster1, cluster2])
    final.to_csv(path_or_buf='submissions/cluster_ensemble' + t.strftime('_%H-%M-%S'), index=False)



