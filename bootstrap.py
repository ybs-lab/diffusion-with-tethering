import numpy as np
from em_algorithm import em_viterbi_optimization
from model import generate_synthetic_trajectories, pack_model_params
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pandas as pd


def generate_theta_hats(N_particles, regime_model_params, random_seed=None, suffix=""):
    """
    Given a set of model parameters, generate N_particles trajectories and calculate their MLE and convergence according to the algorithm (parallelized computation)
    Args:
        N_particles (int): how many trajectories to generate
        regime_model_params: model parameters for generating the trajectories + the initial guess. see pack_model_params() for type
        random_seed (int): seed for generating random trajectories
        suffix: suffix for the new theta hat columns to be added to the df (to distinguish between theta hat vs hat prime)

    Returns:
        df_out (pandas dataframe): contains the true parameter values and the estimated ones for each particle
    """
    T = 10000
    dt = regime_model_params.dt
    N_steps = int(T/dt)
    df_trajs = generate_synthetic_trajectories(N_steps, N_particles, dt, regime_model_params.T_stick,
                                               regime_model_params.T_unstick, regime_model_params.D, regime_model_params.A, random_seed=random_seed)[0]
    all_particles_trajs = [
        x[1][["x", "y"]].values for x in df_trajs.groupby("particle")]
    all_particles_out_df = np.empty(N_particles, dtype=object)

    with ProcessPoolExecutor() as executor:
        for n, output in enumerate(
                executor.map(em_viterbi_optimization, all_particles_trajs,
                             repeat(regime_model_params), repeat(dt))):
            params_by_iter, params_std_by_iter, max_L_by_iter, softmax_by_iter, S_list, X_tether_list, convergence_flag = output
            all_particles_out_df[n] = pd.DataFrame({
                "particle": n,
                "converged": convergence_flag,
                "theta_0_hat"+suffix: params_by_iter[-1][0],
                "theta_1_hat"+suffix: params_by_iter[-1][1],
                "theta_2_hat"+suffix: params_by_iter[-1][2],
                "theta_3_hat"+suffix: params_by_iter[-1][3],
            }, index=[0])
    df_out = pd.concat(all_particles_out_df).reset_index(drop=True)
    return df_out


def analyze_regime_with_bootstrap(T_stick, T_unstick, D, A, dt, T, N_particles, initial_seed=0):
    """
    Given a set of model parameters, do the following:
    1. Generate N_particles trajectories and estimate their MLE theta hat (those particles are called "superparticles")
    2. For each theta hat generate another N_particles trajectories with those specific theta hat and sample their MLE theta hat prime (converged runs only)
    3. Calculate theta hat pprime = 2*theta hat - theta hat prime
    4. Return the df with all the data
    COMMENT: For reproducibility, the random seed for trajectories generation increases by 1 each call to generate_theta_hats().
             Therefore, to avoid repeating the same random seed, have initial_seed to be a product of an integer and some number greate than N_particles + 1.
             In our tests we use N_particles=100 and initial_seed=1000*n_regime
    generate N_particles trajectories and sample their MLE and convergence according to the algorithm (parallelized computation)
    Args:
        T_stick (float): model parameter
        T_unstick (float): model parameter
        D (float): model parameter
        A (float): model parameter
        dt (float): sampling interval, effectively like a model parameter
        T (float): total sampling time
        regime_model_params: model parameters for generating the trajectories + the initial guess. see pack_model_params() for type
        initial_seed (int): initial seed to which new seeds are added
        suffix: suffix for the new theta hat columns to be added to the df (to distinguish between theta hat vs hat prime)

    Returns:
        final_df (pandas dataframe): contains the true parameter values and the estimated ones for each superparticle/particle combination
    """

    true_model_params = pack_model_params(T_stick, T_unstick, D, A, dt)
    random_seed = initial_seed
    df = generate_theta_hats(
        N_particles, true_model_params, random_seed=random_seed)
    df = df[df["converged"]].reset_index(drop=True)
    final_df = pd.DataFrame([])
    for particle in df.particle.unique():
        random_seed += 1
        cur_df = df[df.particle == particle]
        cur_model_params = pack_model_params(
            *[cur_df[f"theta_{i}_hat"].values[0] for i in range(4)], dt)
        cur_out_df = generate_theta_hats(
            N_particles, cur_model_params, random_seed=random_seed, suffix="_prime")
        cur_out_df["superparticle"] = particle
        for i in range(4):
            cur_out_df[f"theta_{i}"] = [T_stick, T_unstick, D, A][i]
            cur_out_df[f"theta_{i}_hat"] = cur_df[f"theta_{i}_hat"].values[0]
            cur_out_df[f"theta_{i}_hat_pprime"] = 2 * \
                cur_out_df[f"theta_{i}_hat"]-cur_out_df[f"theta_{i}_hat_prime"]
        final_df = pd.concat([final_df, cur_out_df])
    return final_df.reset_index(drop=True)
