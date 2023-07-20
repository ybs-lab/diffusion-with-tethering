import numpy as np
from model import pack_model_params, get_optimal_parameters
from viterbi import get_viterbi_paths


def softmax(x):
    """Numerically stable exp(x) / sum(exp(x))"""
    if np.asarray(x).size == 0:
        return np.nan
    else:
        maxVal = x.max()
        exp_arr = np.exp(x - maxVal)
        softmax = exp_arr / np.sum(exp_arr)
        return softmax


def M_step(dt, X_arr, S_list, X_tether_list, L_list):
    """
    This uses analytical formulas to compute the optimal model parameters for complete trajs (including hidden states).
    Args:
        dt: float time step
        X_arr_list: Nx2 array of the particle's positions
        S_list: ndarray of size k_best x N of the k_best most likely hidden state paths S
        X_tether_list: ndarray of size k_best x N x 2 of the k_best most likely tether points
        L_list: len k_best ndarray containing the log-likelihood of each particle's k best hidden paths.
    Returns:
        params_est: ndarray of length 4 with the estimate of the mean of the 4 params (avg over k_best trajectories)
        params_std_est: ndarray of length 4 with the estimate of the std of the 4 params (std over k_best trajectories)
        softmax_mat: len k_best ndarray with the softmax of the log-likelihood of the k_best trajs. it sums to 1.
    """
    params_mat = np.array([get_optimal_parameters(dt, S, X_arr, X_tether)[:4]
                           for S, X_tether in zip(S_list, X_tether_list)])

    keep_inds = np.isfinite(params_mat).all(axis=1)
    params_mat = params_mat[keep_inds]
    softmax_arr = softmax(np.array(L_list[keep_inds]))
    if np.sum(keep_inds) > 0:
        # Now we average over the k best paths
        params_est_mean = softmax_arr @ params_mat
        params_est_std = np.sqrt(softmax_arr @ params_mat ** 2 -
                                 params_est_mean ** 2)  # std = sqrt(E[X^2]-E^2[X])
    else:
        params_est_mean = np.zeros(4) + np.nan
        params_est_std = np.zeros(4) + np.nan

    return params_est_mean, params_est_std, np.array(softmax_arr).flatten()


def em_viterbi_optimization(X_arr, model_params_guess, dt: float,
                            k_best=1, max_iterations=20, stop_threshold=1e-3, verbose=False,
                            pruning_N_states_to_keep=10):
    """
    Args:
        X_arr_list: Nx2 array of the particle's positions
        model_params_guess: initial guess of model parameters for the algorithm. see pack_model_params() for type
        dt: float time interval of equally spaced samples
        k_best: how many best trajectories to take from Viterbi algorithm
        max_iterations: maximal number of iterations (usually there is convergence after 5-6 iters and the loop stops)
        stop_threshold: float threshold for convergence. not very important in most regimes.
        verbose: print output after each E/M step
        pruning_N_states_to_keep: Viterbi pruning parameter, see viterbi.prune_states()

    Returns:
        params_by_iter (N_iter + 1 x 4 ndarray): the 4 model parameters estimated per iteration
        params_std_by_iter (N_iter + 1 x 4 ndarray): the std of the 4 model parameters (based on k_best trajs)
        max_L_by_iter (N_iter x 1 ndarray): the BEST likelihood of the parameters per iteration (for knowledge only)
        softmax_by_iter (N_iter x N_particles x k_best): for each iteration the softmax probability obtained from the
            k best trajectories. It is of interest if this is "flat" or "sharp".
        S_list: ndarray of size k_best x N of the k_best most likely hidden state paths S
        X_tether_list: ndarray of size k_best x N x 2 of the k_best most likely tether points
        is_converged (bool): did the algorithm converge to a solution
    """

    max_L_by_iter = []
    softmax_by_iter = []
    params_by_iter = []
    params_std_by_iter = []
    T = len(X_arr) * dt  # we assume homogenous sampling

    initial_parameters = np.asarray(model_params_guess[:4])  # for documentation
    params_by_iter.append(initial_parameters)
    params_std_by_iter.append(np.zeros(initial_parameters.shape))
    model_params = pack_model_params(*initial_parameters, dt)

    # both flags are needed because it's possible to not converge but without diverging (e.g. surpassing max_iter).
    convergence_flag = False
    divergence_flag = False

    if verbose:
        print("Initial parameters are [{:.2e}, {:.2e}, {:.2e}, {:.2e}]".format(*initial_parameters))

    for iter in range(max_iterations):
        if verbose:
            print(f"Starting iteration {iter} - E step")
        ### E step - find trajectories
        S_list, X_tether_list, L_list = get_viterbi_paths(X_arr, model_params, k_best, pruning_N_states_to_keep)
        # S_list, X_tether_list are arrays of size k_best x N for the k best trajectories

        ### M step - find the parameters
        params_est_mean, params_est_std, softmax_arr = M_step(dt, X_arr, S_list, X_tether_list, L_list)

        if verbose:
            print(f"Done with iteration {iter} - M step")
            print("Current parameter estimates are [{:.2e}, {:.2e}, {:.2e}, {:.2e}]".format(*params_est_mean))

        ### Finish the iteration - log the "by iter" arrays (the other outputs are given in the E step)
        params_by_iter.append(params_est_mean)
        params_std_by_iter.append(params_est_std)
        softmax_by_iter.append(softmax_arr)
        max_L_by_iter.append(L_list[0] / len(X_arr))  # we need this in addition to softmax because softmax sums to 1.
        ### Check for convergence and divergence:

        prev_parameters = params_by_iter[-2]
        # two consecutive divergences will break the loop. divergence = any parameter is not finite.
        if np.any(~np.isfinite(params_est_mean)) or (params_est_mean[0] > 0.9 * T) or (params_est_mean[1] > 0.9 * T):
            if divergence_flag:
                print("Warning! EM algorithm has diverged!")
                break
            else:
                divergence_flag = True
        else:
            divergence_flag = False

        # if the parameters estimate is very close
        if np.max(np.abs(params_est_mean - prev_parameters) / prev_parameters) < stop_threshold:
            convergence_flag = True
            break
        else:
            model_params = pack_model_params(*params_est_mean, dt)
            # and proceed to do another iteration of the loop

    params_by_iter = np.array(params_by_iter)
    params_std_by_iter = np.array(params_std_by_iter)
    max_L_by_iter = np.array(max_L_by_iter)
    softmax_by_iter = np.array(softmax_by_iter)

    return params_by_iter, params_std_by_iter, max_L_by_iter, softmax_by_iter, S_list, X_tether_list, convergence_flag
