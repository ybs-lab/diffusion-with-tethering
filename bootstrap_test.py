import numpy as np
from em_algorithm import em_viterbi_optimization
from model import generate_synthetic_trajectories, pack_model_params
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pandas as pd

def generate_theta_hats(N_particles,regime_model_params,random_seed=None,suffix=""):
    """
    Args:
        model_params_guess: model parameters for generating the trajectories + the initial guess. see pack_model_params() for type
        random_seed: seed for generating random trajectories

    Returns:
        df_out (pandas dataframe): contains the true parameter values and the estimated ones for each particle
    """
    T = 10000
    dt = regime_model_params.dt
    N_steps = int(T/dt)
    df_trajs = generate_synthetic_trajectories(N_steps,N_particles,dt,regime_model_params.T_stick,regime_model_params.T_unstick,regime_model_params.D,regime_model_params.A,random_seed=random_seed)[0]
    all_particles_trajs = [x[1][["x","y"]].values for x in df_trajs.groupby("particle")]
    all_particles_out_df = np.empty(N_particles,dtype=object)
    
    with ProcessPoolExecutor() as executor:
        for n, output in enumerate(
                executor.map(em_viterbi_optimization, all_particles_trajs,
                repeat(regime_model_params),repeat(dt))):
            params_by_iter, params_std_by_iter, max_L_by_iter, softmax_by_iter, S_list, X_tether_list, convergence_flag = output            
            all_particles_out_df[n] = pd.DataFrame({
            "particle": n,
            "converged": convergence_flag,
            "theta_0_hat"+suffix: params_by_iter[-1][0],
            "theta_1_hat"+suffix: params_by_iter[-1][1],
            "theta_2_hat"+suffix: params_by_iter[-1][2],
            "theta_3_hat"+suffix: params_by_iter[-1][3],
            },index=[0])
    df_out = pd.concat(all_particles_out_df).reset_index(drop=True)
    return df_out



def analyze_regime(T_stick,T_unstick,D,A,dt=10,T=10000,N_particles=100,regime_id=0,initial_seed=0):
    true_model_params = pack_model_params(T_stick,T_unstick,D,A,dt)
    random_seed = initial_seed
    df = generate_theta_hats(N_particles,true_model_params,random_seed=random_seed)
    df = df[df["converged"]].reset_index(drop=True)
    final_df = pd.DataFrame([])
    for particle in df.particle.unique():
        random_seed+=1
        cur_df = df[df.particle==particle]        
        cur_model_params = pack_model_params(*[cur_df[f"theta_{i}_hat"].values[0] for i in range(4)],dt)
        cur_out_df = generate_theta_hats(N_particles,cur_model_params,random_seed=random_seed,suffix="_prime")
        cur_out_df["superparticle"] = particle
        for i in range(4):
            cur_out_df[f"theta_{i}"] = [T_stick,T_unstick,D,A][i]
            cur_out_df[f"theta_{i}_hat"] = cur_df[f"theta_{i}_hat"].values[0]
        final_df = pd.concat([final_df,cur_out_df])
    final_df.reset_index(drop=True).to_csv(f"regime_{regime_id}.csv")


if __name__ == '__main__':
    regimes = np.array(
              [[100, 100, 1, 1],
               [200, 50, 1, 1],
               [50, 200, 1, 1],
               [100, 100, 1, 5],
               [100, 100, 1, 10]])
    for n in range(len(regimes)):
        analyze_regime(*regimes[n],regime_id=n,initial_seed=1000*n)
    
        
        