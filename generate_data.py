import os
import numpy as np
from tests import accuracy_of_hidden_path
from em_algorithm import em_viterbi_optimization
from model import pack_model_params, generate_synthetic_trajectories, get_optimal_parameters
from scipy.stats import loguniform
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pandas as pd


def generate_regimes_table():
    dt_list = [10., 1., 0.5, 10., 10., 10, 10]
    T_stick_list = [100., 100., 100., 50., 20., 200, 50]
    T_unstick_list = [100., 100., 100., 50., 20., 50, 200]
    regimes_dict = {}
    for i in range(7):
        regimes_dict[i] = {"dt": dt_list[i], "T_stick": T_stick_list[i], "T_unstick": T_unstick_list[i]}
    return regimes_dict


def test_indifference_to_initial_condition(N_init_conditions, selected_regime=0):
    regimes = generate_regimes_table()
    D = 1.
    A = 1.
    T_stick = regimes[selected_regime]["T_stick"]
    T_unstick = regimes[selected_regime]["T_unstick"]
    dt = regimes[selected_regime]["dt"]
    T = 10000
    params_arr = np.array([T_stick, T_unstick, D, A])
    N_steps = int(T / dt)
    df, model_params = generate_synthetic_trajectories(N_steps, 1, dt, T_stick, T_unstick, D, A, random_seed=42)
    X_arr = df[["x", "y"]].values
    th0_ratio_arr = loguniform.rvs(1. / 10, 10., size=(N_init_conditions, 4))
    th0 = 0 * th0_ratio_arr
    for i in range(N_init_conditions):
        th0[i] = th0_ratio_arr[i] * [T_stick, T_unstick, D, A]  # theta 0
    init_model_params_list = []
    for i in range(N_init_conditions):
        init_model_params_list.append(pack_model_params(*th0[i], dt))
    params_list_arr = np.empty(N_init_conditions, dtype=object)
    converged_arr = np.zeros(N_init_conditions)

    with ProcessPoolExecutor(max_workers=20) as executor:
        for i, output in enumerate(executor.map(em_viterbi_optimization,
                                                repeat(X_arr), init_model_params_list, repeat(dt),
                                                repeat(1),
                                                )):
            converged_arr[i] = output[6]
            params_list_arr[i] = output[0]  # output params
            print(i, flush=True)

    converged_arr = converged_arr.astype(bool)
    max_iters = np.max([len(params_list) for params_list in params_list_arr])
    theta_mat = np.zeros([N_init_conditions, max_iters, 4])
    for i in range(N_init_conditions):
        params_list = params_list_arr[i]  # nx4
        last_params = params_list[-1]
        rows_to_fill = max_iters - len(params_list)
        if rows_to_fill > 0:
            theta_mat[i] = np.vstack(
                [params_list, np.array([last_params for _ in range(max_iters - len(params_list))])])
        else:
            theta_mat[i] = params_list

    optimal_model_params = get_optimal_parameters(dt, df.state.values, X_arr)
    optimal_params = np.array(
        [optimal_model_params.T_stick, optimal_model_params.T_unstick, optimal_model_params.D,
         optimal_model_params.A])

    print(f"Convergence rate: {np.mean(converged_arr):.2f}")

    os.makedirs('./Data', exist_ok=True)
    df.to_csv(f"./Data/indifference_to_init_conditions_df_regime_{selected_regime}.csv")
    np.save(f"./Data/indifference_to_init_conditions_params_list_arr_regime_{selected_regime}.npy", params_list_arr)
    np.save(f"./Data/indifference_to_init_conditions_converged_arr_regime_{selected_regime}.npy", converged_arr)
    np.save(f"./Data/indifference_to_init_conditions_optimal_params_regime_{selected_regime}.npy", optimal_params)


def test_accuracy(N_realizations, regimes_arr=np.arange(7, dtype=int)):
    N_regimes = len(regimes_arr)
    params_arr = np.zeros([N_regimes, N_realizations, 4])
    accuracy_arr = np.zeros([N_regimes, N_realizations])
    converged_arr = np.zeros([N_regimes, N_realizations], dtype=bool)
    N_iters_arr = np.zeros([N_regimes, N_realizations], )

    T = 10000.
    D = 1.
    A = 1.
    regimes = generate_regimes_table()
    for i, reg in enumerate(regimes_arr):
        T_stick = regimes[reg]["T_stick"]
        T_unstick = regimes[reg]["T_unstick"]
        dt = regimes[reg]["dt"]
        N_steps = int(T / dt)

        df, model_params = generate_synthetic_trajectories(N_steps, N_realizations, dt, T_stick, T_unstick, D, A, )
        X_arr_list = [df_particle[["x", "y"]].values for _, df_particle in df.groupby("particle")]
        S_list = [df_particle["state"].values for _, df_particle in df.groupby("particle")]
        X_tether_list = [df_particle[["x_tether", "y_tether"]].values for _, df_particle in df.groupby("particle")]

        with ProcessPoolExecutor(max_workers=20) as executor:
            for j, output in enumerate(executor.map(em_viterbi_optimization, X_arr_list,
                                                    repeat(model_params),
                                                    repeat(dt))):
                converged_arr[i, j] = output[6]
                params_arr[i, j] = output[0][-1]  # output params
                accuracy_arr[i, j] = accuracy_of_hidden_path(output[4][0], output[5][0], S_list[j],
                                                             X_tether_list[j])
                N_iters_arr[i, j] = len(output[0]) - 1
                print(f"{j}, {output[0][-1][0]}, {output[0][-1][1]}, {output[0][-1][2]}, {output[0][-1][3]}",
                      flush=True)
    df = pd.DataFrame([])
    for n, reg in enumerate(regimes_arr):
        cur_df = pd.DataFrame({
            "regime": reg,
            "realization": np.arange(N_realizations),
            "converged": converged_arr[n],
            "accuracy": accuracy_arr[n],
            "iters": N_iters_arr[n],
            "T0": params_arr[n, :, 0],
            "T1": params_arr[n, :, 1],
            "D": params_arr[n, :, 2],
            "A": params_arr[n, :, 3],
            "dt": regimes[reg]["dt"],
            "tau0_true": regimes[reg]["T_stick"],
            "tau1_true": regimes[reg]["T_unstick"],
            "T": 10000,
        })
        df = pd.concat([df, cur_df]).reset_index(drop=True)
    os.makedirs('Data', exist_ok=True)
    df.to_csv("./Data/test_accuracy_df_full.csv")


def K_most_likely_data():
    D, A, dt, T_stick, T_unstick, T = [1, 1, 10, 100, 100, 10000]
    df, model_params = generate_synthetic_trajectories(int(T / dt), 1, dt, T_stick, T_unstick, D, A, random_seed=8)
    df = df[:300].reset_index(drop=True)
    # oracle parameters
    model_params = get_optimal_parameters(dt, df.state.values, df[["x", "y"]].values,
                                          df[["x_tether", "y_tether"]].values)

    os.makedirs('Data', exist_ok=True)
    df.to_csv("./Data/k_most_likely_orig_df.csv")

    output = em_viterbi_optimization(df[["x", "y"]].values, model_params, dt=dt,
                                     verbose=True, k_best=20, stop_threshold=1e-3)
    np.save("./Data/K_most_likely_alg.npy", np.array(output,dtype=object))

    output = em_viterbi_optimization(df[["x", "y"]].values, model_params, dt=dt,
                                     verbose=True, k_best=20, stop_threshold=1e3)
    np.save("./Data/K_most_likely_oracle.npy", np.array(output,dtype=object))


def generate_traj_for_fig2():
    D, A, dt, T_stick, T_unstick, T = [1, 0.5, 1, 100, 100, 2000]
    N_steps = int(T / dt)
    df, _ = generate_synthetic_trajectories(N_steps, 1, dt, T_stick, T_unstick, D, A,random_seed=0)
    os.makedirs('Data', exist_ok=True)
    df.to_csv("./Data/example_traj_df.csv")
