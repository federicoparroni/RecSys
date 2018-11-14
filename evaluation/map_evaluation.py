def evaluate_map(recc, test_urm, at=10):
    """
    gives the map(at) evaluation of the recc wrt the test_urm
    @params:
        recc        - (list) reccomendations that come out from model_bridge
        test_urm    - (csr_matrix) urm with all the playlists and the songs that were masked during training time
        at          - (int) param of map, map(at)
    """
    aps = []
    for i in recc:
        row = test_urm.getrow(i[0]).indices
        m = min(at, len(row))

        ap = 0
        for j in range(1, m+1):
            n_elems_found = 0
            if i[j] in row:
                n_elems_found += 1
                ap += n_elems_found/j
        if m:
            ap /= m
            aps.append(ap)

    return sum(aps)/len(recc)
