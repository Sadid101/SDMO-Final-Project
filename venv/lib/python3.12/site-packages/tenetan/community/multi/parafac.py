from tenetan.networks import SnapshotGraph
import tensorly as tl
import tlviz

import logging

__all__ = ['PARAFAC_NN_ALS']


def PARAFAC_NN_ALS(network: SnapshotGraph, n_communities: int, /, *to_return, **tensorly_kwargs):
    """
    Temporal Community detcetion via Non-negative Alternating Least Squares PARAFAC Decomposition
    [1] Gauvin L, Panisson A, Cattuto C (2014) Detecting the Community Structure and Activity Patterns of Temporal Networks:
    A Non-Negative Tensor Factorization Approach. PLoS ONE 9(1): e86028.
    https://doi.org/10.1371/journal.pone.0086028
    :param network: Temporal network
    :param n_communities: Number of communities to detect
    :param to_return: Data to return, can container the following fields:
        - "in_communities": matrix describing the in-membership of the nodes to communities (first tensor of decomposition)
        - "out_communities": matrix describing the out-membership of the nodes to communities (second tensor of decomposition)
        - "raw_temporal_activity": matrix describing the temporal activity of communities (third tensor of decomposition)
        - "in_temporal_activity": raw temporal activity scaled by the sum of all nodes' in_community weights in that community (eq. 9 in [1])
        - "out_temporal_activity": raw temporal activity scaled by the sum of all nodes' out_community weights in that community (eq. 9 in [1])
        - "core_consistency": core consistency metric of the calculated decomposition
        - "errors": errors of decomposition of each iteration as return by tensorly.decomposition.non_negative_pafarac_hals
    :param tensorly_kwargs: kwargs to pass to tensorly.decomposition.non_negative_parafac_hals
    :return: a dict of requested data, default ('in_communities', 'out_communities', 'raw_temporal_activity')
    """

    if len(to_return) == 0:
        to_return = ('in_communities', 'out_communities', 'raw_temporal_activity')
    tensorly_kwargs['return_errors'] = True

    tensor = tl.tensor(network.tensor)
    cp_tensor, errors = tl.decomposition.non_negative_parafac_hals(tensor, n_communities, **tensorly_kwargs)
    in_communities, out_communities, raw_temporal_activity = cp_tensor.factors

    #  Construct output
    return_dict = {}
    for data in to_return:  # TODO Switch to match when upgrading to 9.10
        if data == "in_communities":
            return_dict[data] = in_communities
        elif data == "out_communities":
            return_dict[data] = out_communities
        elif data == "raw_temporal_activity":
            return_dict[data] = raw_temporal_activity
        elif data == "in_temporal_activity":  # TODO add proper matrix handling instead of the for-loops
            weights = tl.sum(in_communities, axis=0)
            in_temporal_activity = tl.zeros_like(raw_temporal_activity)
            for i in range(tl.shape(in_temporal_activity)[0]):
                for j in range(tl.shape(in_temporal_activity)[1]):
                    in_temporal_activity = tl.index_update(in_temporal_activity, tl.index[i, j],
                                                           raw_temporal_activity[i, j] * weights[j])
            return_dict[data] = in_temporal_activity
        elif data == "out_temporal_activity":
            weights = tl.sum(out_communities, axis=0)
            out_temporal_activity = tl.zeros_like(raw_temporal_activity)
            for i in range(tl.shape(out_temporal_activity)[0]):
                for j in range(tl.shape(out_temporal_activity)[1]):
                    out_temporal_activity = tl.index_update(out_temporal_activity, tl.index[i, j],
                                                            raw_temporal_activity[i, j]*weights[j])
            return_dict[data] = out_temporal_activity
        elif data == 'core_consistency':
            return_dict[data] = tlviz.model_evaluation.core_consistency(cp_tensor, network._tensor)
        elif data == "errors":
            return_dict[data] = errors
        else:
            logging.error(f"{__name__}: unknown data requested from parafac_communities - {data}")

    return return_dict
