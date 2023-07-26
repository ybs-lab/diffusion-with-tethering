import numpy as np
import pandas as pd
from viterbi import viterbi_paths_to_end_nodes, get_viterbi_paths
from model import pack_model_params, get_optimal_parameters, generate_synthetic_trajectories
from em_algorithm import em_viterbi_optimization
import unittest
import cProfile
import pstats
import io


def profile(mode, profiler=[], filename='profile_results'):
    if mode.lower() == "on":
        pr = cProfile.Profile()
        pr.enable()
        return pr
    elif mode.lower() == "off" or mode.lower() == "viewer":
        pr = profiler
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        with open((filename + '.txt'), 'w+') as f:
            f.write(s.getvalue())
        pr.dump_stats((filename + '.prof'))
        print(
            "For a good ui view, execute in terminal: snakeviz ./" + filename + ".prof" + "\nOr go to https://nejc.saje.info/pstats-viewer.html")


def accuracy_of_hidden_path(S_pred, X_tether_pred, S_true, X_tether_true):
    accuracy = np.mean((S_pred == 0) * (S_true == 0) + (S_pred == 1) * (S_true == 1) * np.prod(
        (X_tether_pred == X_tether_true), 1))
    return accuracy


class TestViterbiAlgorithm(unittest.TestCase):
    def test_get_viterbi_paths(self):
        D = 1.
        A, dt, T_stick, T_unstick, T = [1, 10, 2 * 10 ** 3, 2 * 10 ** 3, 10 ** 5]
        A *= D
        N_steps = int(T / dt)
        df, model_params = generate_synthetic_trajectories(N_steps, 1, dt, T_stick, T_unstick, D, A)
        X_arr = df[["x", "y"]].values

        # p = profile("on")
        S_arr, X_tether_arr, viterbi_paths_log_likelihood = get_viterbi_paths(X_arr, model_params, 10, 5)
        # profile("viewer", p)

        S_best = S_arr[0].reshape(N_steps)
        X_tether_best = X_tether_arr[0].reshape((N_steps, 2))
        accuracy = accuracy_of_hidden_path(S_best, X_tether_best, df.state.values, df[["x_tether", "y_tether"]].values)
        assert accuracy > 0.95  # 95% accuracy at least in this easy regime


class TestGetOptimalParameters(unittest.TestCase):
    def test_get_optimal_parameters(self):
        D = 1.
        A, dt, T_stick, T_unstick, T = [1, 10, 2 * 10 ** 3, 2 * 10 ** 3, 10 ** 5]
        A *= D
        N_steps = int(T / dt)
        df, model_params = generate_synthetic_trajectories(N_steps, 1, dt, T_stick, T_unstick, D, A,random_seed=0)

        params_est = get_optimal_parameters(dt, df.state.values, df[["x", "y"]].values,
                                            df[["x_tether", "y_tether"]].values)
        params_est = np.array([
            params_est.T_stick, params_est.T_unstick, params_est.D, params_est.A,
        ])
        rel_std_allowed_arr = np.array([
            0.5,
            0.5,
            0.25,
            0.25]
        )
        params_true = np.array([T_stick, T_unstick, D, A])
        assert np.all(np.abs(params_est - params_true) / params_true < rel_std_allowed_arr)


class TestEM(unittest.TestCase):
    def test_em_viterbi_optimization(self):
        D = 1.
        A, dt, T_stick, T_unstick, T = [1, 10, 2 * 10 ** 3, 2 * 10 ** 3, 10 ** 5]
        A *= D
        N_steps = int(T / dt)
        df, model_params = generate_synthetic_trajectories(N_steps, 1, dt, T_stick, T_unstick, D, A)

        X_arr = df[["x", "y"]].values
        model_params = pack_model_params(T_stick * 2, T_unstick / 2, D / 2, A * 2, dt)
        params_by_iter, params_std_by_iter, max_L_by_iter, softmax_by_iter, S_list, X_tether_list, convergence_flag = \
            em_viterbi_optimization(X_arr, model_params, dt, k_best=1, verbose=False)
        params_est = params_by_iter[-1]

        # first [0] is for n_particle, second [0] is for best out of k_best
        accuracy = accuracy_of_hidden_path(S_list[0], X_tether_list[0],
                                           df["state"].values, df[["x_tether", "y_tether"]].values)

        rel_std_allowed_arr = np.array([
            1.,  # IMPROVE THIS LATER WITH INVERSE POISSON T/T_stick ETC
            1.,
            0.1,
            0.1]
        )
        params_true = np.array([T_stick, T_unstick, D, A])
        assert np.all(np.abs(params_est - params_true) / params_true < rel_std_allowed_arr)
        assert accuracy > 0.9  # 90% accuracy
        assert convergence_flag

if __name__ == '__main__':
    print('Running all tests.')
    unittest.main()
