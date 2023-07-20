import numpy as np
from model import transition_log_probability


class Path:
    """
    Class Path: is used in the viterbi_paths_to_end_nodes function
    The attributes are log-likelihood L and nodes which is a tuple of ints.
    The methods except for extends allow for comparison of likelihoods of different paths.
    """

    def __init__(self, nodes: tuple, L=0.):
        self.nodes = nodes  # nodes is a tuple of varying length, each entry is a node.
        self.L = L

    def extend(self, node, delta_L):
        # Append a new node to nodes and add the delta log-likelihood
        return Path(self.nodes + (node,), self.L + delta_L)

    def __ge__(self, x):
        return self.L.__ge__(x.L)

    def __le__(self, x):
        return self.L.__le__(x.L)

    def __gt__(self, x):
        return self.L.__gt__(x.L)

    def __lt__(self, x):
        return self.L.__lt__(x.L)

    def __eq__(self, x):
        return self.L.__eq__(x.L)

    def __repr__(self):
        return f"Path({self.nodes}, {self.L})"


def edge_transition_log_likelihood(i: np.ndarray, j: int, n: int, X_arr: np.ndarray, model_params):
    """
    :param i: ndarray of SOURCE hidden states
    :param j: TARGET hidden state
    :param n: TARGET time
    :param X_arr: Nx2 ndarray with all the observed particle's positions
    :param model_params: Params namedtuple, see model.pack_model_params()
    :return: L which is an ndarray of the same shape as i, represents the log-likelihood of i->j transitions
    """
    X_prev = X_arr[[n - 1]]  # need X_arr[ [n-1] ] to keep 1x2 shape
    X_curr = X_arr[[n]]
    X_tether_prev = X_arr[i - 1]  # i is an array so it's ok like that

    L = transition_log_probability(i, j, X_prev, X_curr, X_tether_prev, model_params)
    return L


def prune_states(paths_to_nodes, possible_target_states_mask, pruning_N_states_to_keep):
    """
    Args:
        paths_to_nodes: np.ndarray of lists of current paths to all nodes - *this is updated in the function*
        possible_target_states_mask: mask of allowed target states - *this is updated in the function*
        pruning_N_states_to_keep: *important parameter* of how many stuck states to keep.
    This function keeps the pruning_N_states_to_keep most likely stuck states, for performance purposes.
    """
    paths_to_nodes = paths_to_nodes.copy()  # copy so we won't edit the input array (for readability)
    possible_target_states_mask_without_free = possible_target_states_mask.copy()
    possible_target_states_mask_without_free[0] = False

    # prune available stuck states
    best_L_of_states = [np.max(path_arr).L for path_arr in paths_to_nodes[possible_target_states_mask_without_free]]
    inds_to_keep = np.where(possible_target_states_mask_without_free)[0][
        np.flip(np.argsort(best_L_of_states))[:pruning_N_states_to_keep]]

    # create the mask again
    possible_target_states_mask = False * possible_target_states_mask
    possible_target_states_mask[inds_to_keep] = True
    possible_target_states_mask[0] = True  # free state is always available

    # the next line is important for memory purposes - stop storing paths for states we discarded
    paths_to_nodes[~possible_target_states_mask] = Path((0,), -np.inf)  # clear all states we discarded
    return paths_to_nodes, possible_target_states_mask


def viterbi_paths_to_end_nodes(X_arr, model_params, k_best=1, pruning_N_states_to_keep=10):
    """
    Implementation of the Viterbi algorithm that returns the k most likely trajectories on the trellis graph.
    Args:
        X_arr: Nx2 ndarray with all the observed particle's positions
        model_params: Params namedtuple, see model.pack_model_params()
        k_best: how many most likely viterbi paths to the end nodes to keep
        pruning_N_states_to_keep: keep this small for performance, see prune_states()
    Returns:
        paths_to_nodes - list of length len(X_arr)+1 [per end node] that contains at element n another list of
        the k_best most likely Paths (Path class) to end node n. Practically this is very sparse and contains only up to
        pruning_N_states_to_keep+1 nonempty lists.
    """
    N_steps = len(X_arr)
    N_states = N_steps + 1
    prev_paths_to_nodes = np.empty(N_states, dtype=list)
    paths_to_nodes = np.empty(N_states, dtype=list)
    possible_target_states_mask = np.zeros(N_states, dtype=bool)

    # initialization
    for j in range(2):
        # zero log-likelihood at time zero - this can be modified for a more accurate prior
        prev_paths_to_nodes[j] = [Path((j,), 0.)]
        possible_target_states_mask[j] = True

    # loop over the column n (time) in our trellis diagram
    for n in range(1, N_steps):
        if n > 1:
            # prune available stuck states (not at the first iteration)
            paths_to_nodes, possible_target_states_mask = prune_states(paths_to_nodes,
                                                                       possible_target_states_mask,
                                                                       pruning_N_states_to_keep)
            # update previous paths to nodes (not at the first iteration)
            prev_paths_to_nodes = paths_to_nodes.copy()

        possible_target_states_mask[n + 1] = True  # extend available states one at a time
        possible_target_states = np.where(possible_target_states_mask)[0]

        for j in possible_target_states:
            # The free state can have sources from all previous states.
            if j == 0:
                possible_source_states = possible_target_states[:-1]
            # The next stuck state can be accessed only from the free state.
            elif j == (n + 1):
                possible_source_states = np.array([0])
            # Any previous stuck state can only be accessed from itself .
            else:
                possible_source_states = np.array([j])

            edge_transitions = edge_transition_log_likelihood(possible_source_states, j, n, X_arr,
                                                              model_params)  # vectorized on possible_source_states
            new_paths = np.hstack(  # these are all paths to node j
                [
                    [
                        path.extend(j, edge_transitions[n_source_state])
                        for path in prev_paths_to_nodes[i]  # inner loop: on paths to node i
                    ]
                    for n_source_state, i in enumerate(possible_source_states)  # outer loop: on i in possible sources
                ])

            # this assignment is why we need prev_paths_to_nodes as well - otherwise this will interfere with new_paths definition
            paths_to_nodes[j] = sorted(new_paths, reverse=True)[:k_best]  # for j>0 this is a singleton

    return paths_to_nodes


def get_viterbi_paths(X_arr, model_params, k_best, pruning_N_states_to_keep):
    """
    This calls viterbi_paths_to_end_nodes() and gets the k most likely paths (regardless of end node)
    Args:
        X_arr: Nx2 ndarray with all the observed particle's positions
        model_params: Params namedtuple, see model.pack_model_params()
        k_best: return the k_best most likely paths
    Returns:
        S_arr - (k_best x N) ndarray of the hidden states S of the k_best likeliest paths
        X_tether_arr - (k_best x N x 2) ndarray of the tether point of the k_best likeliest paths
        viterbi_paths_log_likelihood - the log-likelihood of the k_best likeliest paths
    """
    paths_to_end_nodes = viterbi_paths_to_end_nodes(X_arr, model_params, k_best=k_best,
                                                    pruning_N_states_to_keep=pruning_N_states_to_keep)
    k_best_paths = sorted(np.hstack(paths_to_end_nodes), reverse=True)[:k_best]

    S_arr = np.zeros([k_best, len(X_arr)])
    X_tether_arr = np.zeros([k_best, len(X_arr), 2])
    viterbi_paths_log_likelihood = np.zeros(k_best)

    for l in range(k_best):
        hidden_states = np.array(k_best_paths[l].nodes)
        S_arr[l] = np.int_(hidden_states > 0)
        X_tether_arr[l] = X_arr[hidden_states - 1]
        viterbi_paths_log_likelihood[l] = k_best_paths[l].L
    X_tether_arr[S_arr == 0] += np.nan  # X_tether is nan where free (for clarity purposes)
    return S_arr, X_tether_arr, viterbi_paths_log_likelihood
