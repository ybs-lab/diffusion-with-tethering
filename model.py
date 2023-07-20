import numpy as np
from numpy import random
import pandas as pd
from enum import Enum
from collections import namedtuple
import numba

# See pack_model_params function for explanations regarding this namedtuple
Params = namedtuple('Params',
                    ['T_stick', 'T_unstick', 'D', 'A', 'dt',
                     'MSD', 'log_pi_MSD', 'inertia_factor', 'modified_2A', 'log_pi_modified_2A',
                     'log_stay_free', 'log_stick', 'log_unstick', 'log_stay_stuck', ])


# for generating trajectories
class GenerationMode(Enum):
    DONT_FORCE = 0
    FORCE_FREE = 1
    FORCE_STUCK = 2


def pack_model_params(T_stick: float, T_unstick: float, D: float, A: float, dt: float):
    """
    Pack a namedtuple containing all the model parameters, and also all the derived parameters used for the calculation
    (e.g. calculate the log of some parameters here ones, instead of computing log each time)
    This is namedtuple instead of dict for compatibility with numba
    (see https://stackoverflow.com/questions/46003172/replacement-of-dict-type-for-numba-as-parameters-of-a-python-function)
    """
    d = 2  # spatial dimension
    MSD = 4 * D * dt  # comment: this is just a name for 4D\Delta t. in d!=2 the real MSD is (d/2)*4D\Delta t.
    r = 1. / T_stick + 1. / T_unstick  # combined rate of transitions

    inertia_factor = np.exp(-D * dt / A)  # important factor for the stuck state
    modified_2A = 2 * A * (1 - inertia_factor ** 2)  # 2\tilde{A} in the paper

    log_pi_MSD = 0.5 * d * np.log(np.pi * MSD)
    log_pi_modified_2A = 0.5 * d * np.log(np.pi * modified_2A)

    # log of temporal parameters - from the exact solution of a 2 state continuous time Markov chain
    phi = np.exp(-r * dt)
    log_stay_free = np.log((T_stick + T_unstick * phi) / (T_stick + T_unstick))
    log_stay_stuck = np.log((T_unstick + T_stick * phi) / (T_stick + T_unstick))
    log_stick = np.log((T_unstick * (1 - phi)) / (T_stick + T_unstick))
    log_unstick = np.log((T_stick * (1 - phi)) / (T_stick + T_unstick))

    model_params = Params(
        T_stick=T_stick,
        T_unstick=T_unstick,
        D=D,
        A=A,
        dt=dt,
        MSD=MSD,
        log_pi_MSD=log_pi_MSD,
        inertia_factor=inertia_factor,
        modified_2A=modified_2A,
        log_pi_modified_2A=log_pi_modified_2A,
        log_stay_free=log_stay_free,
        log_stick=log_stick,
        log_unstick=log_unstick,
        log_stay_stuck=log_stay_stuck,
    )
    return model_params


@numba.jit(nopython=True) #this is numbaed because it is called many times at each Viterbi run
def transition_log_probability(S_prev, S_curr, X_prev, X_curr, X_tether_prev, model_params):
    """
    Calculate the log of probability of transition from one state to another, when they are separated by time dt, using
    the model parameters. In essence this incorporates all the model information.

    Args: [M is 1 or number of transitions for vectorized implementation thanks to broadcast)]
        S_prev (M array of int): 0 for free, 1 for stuck, of the state the transition is FROM
        S_curr (M array of int): 0 for free, 1 for stuck, of the state the transition is TO
        X_prev (Mx2 float): position (x,y) of the state the transition is FROM
        X_curr (Mx2 float): position (x,y) of the state the transition is TO
        X_tether_prev (Mx2 float): tether position (x,y) of the state the transition is FROM
        model_params: this includes all the model parameters, the time step dt, and some products and logs of them
        (comment: don't need X_tether_curr)

    Returns:
        L (M array of float): log of the probability of the transition. Could be -np.inf for zero-probability transitions.
    """

    # Now allows for vectorized implementations

    # bool array
    prev_free = (S_prev == 0)
    prev_stuck = np.logical_not(prev_free)
    curr_free = (S_curr == 0)
    curr_stuck = np.logical_not(curr_free)

    # Addition is done here because of broadcasting in axis 1
    dX_free = (X_curr - X_prev)
    dX_stuck = (X_curr - X_tether_prev) - model_params.inertia_factor * (X_prev - X_tether_prev)
    spatial_free = - np.sum(dX_free ** 2, axis=1) / model_params.MSD - model_params.log_pi_MSD
    spatial_stuck = - np.sum(dX_stuck ** 2, axis=1) / model_params.modified_2A - model_params.log_pi_modified_2A

    spatial = prev_free * spatial_free + prev_stuck * spatial_stuck
    temporal = prev_free * (curr_free * model_params.log_stay_free + curr_stuck * model_params.log_stick) + \
               prev_stuck * (curr_free * model_params.log_unstick + curr_stuck * model_params.log_stay_stuck)
    L = spatial + temporal

    return L


def get_optimal_parameters(dt, S_arr, X_arr, X_tether_arr=None):
    """
    Calculate the most likely model parameters given a full trajectory

    Args:
        dt (float): time interval
        S_arr (Nx1 int): list of states - 0 is free, 1 is stuck
        X_arr (Nx2 float): list of particle positions
        X_tether_arr (Nx2 float): list of particle tether positions. optional argument (can be derived from the others)

    Returns:
        Params namedtuple (see pack_model_params)
    """

    if X_tether_arr is None:  # we can calculate X_tether from S and X
        X_tether_arr = calc_x_tether(S_arr, X_arr)

    if not S_arr.any():  # completely free trajectory
        T_stick_est = np.inf
        T_unstick_est = np.nan
        D_est = np.sum(np.diff(X_arr, axis=0) ** 2, axis=1).mean(axis=0) / (4 * dt)
        A_est = np.nan
    elif S_arr.all():  # completely stuck trajectory
        T_stick_est = np.inf
        T_unstick_est = np.nan
        D_est = np.nan
        A_est = np.nanmean(np.sum((X_arr[1:] - X_tether_arr[:-1]) ** 2, axis=1), axis=0) / 2
    else:  # mix of states
        mask_free_to_free = np.logical_and(S_arr[:-1] == 0, S_arr[1:] == 0)
        mask_free_to_stuck = np.logical_and(S_arr[:-1] == 0, S_arr[1:] != 0)
        mask_stuck_to_free = np.logical_and(S_arr[:-1] != 0, S_arr[1:] == 0)
        mask_stuck_to_stuck = np.logical_and(S_arr[:-1] != 0, S_arr[1:] != 0)

        mask_free = np.logical_not(S_arr[:-1].astype('bool'))
        mask_stuck = S_arr[:-1].astype('bool')

        # wrapped with if because of pesky warnings
        if mask_free_to_stuck.any():
            T_stick_est = (1 + np.sum(mask_free_to_free) / np.sum(mask_free_to_stuck)) * dt
        else:
            T_stick_est = np.inf
        if mask_stuck_to_free.any():
            T_unstick_est = (1 + np.sum(mask_stuck_to_stuck) / np.sum(mask_stuck_to_free)) * dt
        else:
            T_unstick_est = np.inf
        D_est = np.sum((mask_free * np.sum(np.diff(X_arr, axis=0) ** 2, axis=1))) / (4 * dt) / np.sum(mask_free)
        A_est = np.nansum((mask_stuck * np.sum((X_arr[1:] - X_tether_arr[:-1]) ** 2, axis=1))) / 2 / np.sum(
            mask_stuck)

    model_params = pack_model_params(T_stick_est, T_unstick_est, D_est, A_est, dt)
    return model_params


def generate_trajectories(N_steps: int, N_particle: int, init_S, model_params: Params,
                                generation_mode=GenerationMode.DONT_FORCE, random_seed=None):
    """
    Generate a series of States drawn from the distribution corresponding to the model. This has a vectorized
    implementation for generating trajectories of multiple particles. All particles have the same trajectory length.

    Args:
        N_steps: duration of trajectory (in steps) for each of the particles
        N_particles: how many trajectories to generate. Note: all trajectories start at X=[0,0].
        init_S: initial S for each of the particles: 0 is free, 1 is stuck, None is random for each particle (50%).
        generation_mode: FORCE_FREE and FORCE_STUCK make all the particles free or stuck all the time. DONT_FORCE
        allows for transitions according to the model.
        model_params: Params namedtuple (see pack_model_params)
        random_seed: random seed for generating the trajectory

    Returns:
        states_arr (N_particle x N_steps int ndarray): states for each particle at each time step.
        X_arr (N_particle x N_steps x 2 float ndarray): positions (x,y) for each particle at each time step.
        X_tether_arr (N_particle x N_steps x 2 float ndarray): tether point (x,y) for each particle at each step.
        (Comment: effectively this is a N_particle x N_step State matrix.)

    """

    rng = np.random.default_rng(seed=random_seed)  # default is None i.e. random seed
    d = 2  # dimension
    if generation_mode == GenerationMode.FORCE_FREE:
        init_S = 0
    elif generation_mode == GenerationMode.FORCE_STUCK:
        init_S = 1

    if init_S == 0:
        init_state_arr = np.zeros(N_particle, dtype='i')
    elif init_S == 1:
        init_state_arr = np.ones(N_particle, dtype='i')
    elif init_S is None:
        # here we do 50/50 for the initial state even though the rigorous thing is to do S=1 with T_unstick / (T_stick+T_unstick)
        init_state_arr = rng.integers(low=0, high=1, size=N_particle).astype(int)
    else:
        raise RuntimeError(f'Input init_S must be 0, 1 or None. received: {init_S}')

    states_arr = np.zeros([N_particle, N_steps], dtype=int)
    X_arr = np.zeros([N_particle, N_steps, d])
    X_tether_arr = np.zeros([N_particle, N_steps, d])
    states_arr[:, 0] = init_state_arr

    P = np.zeros([2, 2])
    if generation_mode == GenerationMode.DONT_FORCE:
        # note: this is valid only when dt<<T_stick,T_unstick
        P[0, 0] = np.exp(model_params.log_stay_free)
        P[0, 1] = np.exp(model_params.log_stick)
        P[1, 0] = np.exp(model_params.log_unstick)
        P[1, 1] = np.exp(model_params.log_stay_stuck)
    if generation_mode == GenerationMode.FORCE_FREE:
        P[:, 0] = 0.
        P[:, 1] = 1.
    elif generation_mode == GenerationMode.FORCE_STUCK:
        P[:, 0] = 1.
        P[:, 1] = 0.

    # Stream of 2D gaussian RV with variance 1
    gaussian_Stream = rng.normal(loc=0.0, scale=1, size=[N_particle, N_steps, d])

    uniform_Stream = rng.random(size=[N_particle, N_steps])

    # for clarity X_tether is initialized only for stuck
    X_tether_arr[np.where(init_state_arr == 1), 0] = 0.

    for n in range(1, N_steps):
        free_inds = np.where(states_arr[:, n - 1] == 0)[0]
        stuck_inds = np.where(states_arr[:, n - 1] == 1)[0]

        # Free particles diffuse
        X_arr[free_inds, n] = X_arr[free_inds, n - 1] + \
                              np.sqrt(0.5 * model_params.MSD) * gaussian_Stream[free_inds, n]
        X_tether_arr[free_inds, n] = np.nan

        # Stuck particles wiggle
        X_arr[stuck_inds, n] = X_tether_arr[stuck_inds, n - 1] * (1 - model_params.inertia_factor) + \
                               X_arr[stuck_inds, n - 1] * model_params.inertia_factor + \
                               np.sqrt(0.5 * model_params.modified_2A) * gaussian_Stream[stuck_inds, n]

        # Tether point continues UNLESS going to stick
        X_tether_arr[:, n] = X_tether_arr[:, n - 1]

        # Stick or unstick:
        mask_stick = uniform_Stream[free_inds, n] > P[0, 0]
        mask_unstick = uniform_Stream[stuck_inds, n] > P[1, 1]
        sticking_inds = free_inds[mask_stick]
        staying_free_inds = free_inds[~mask_stick]
        unsticking_inds = stuck_inds[mask_unstick]
        staying_stuck_inds = stuck_inds[~mask_unstick]

        states_arr[np.union1d(unsticking_inds, staying_free_inds), n] = 0
        states_arr[np.union1d(sticking_inds, staying_stuck_inds), n] = 1

        # Sticking particles tether to a new point
        X_tether_arr[sticking_inds, n] = X_arr[sticking_inds, n]

    return states_arr, X_arr, X_tether_arr


def trajectory_log_probability(S_arr, X_arr, model_params):
    """
    Calculate the log of probability of one particle's trajectory (including hidden states)

    Args:
        S_arr (Nx1 int): list of states - 0 is free, 1 is stuck
        X_arr (Nx2 float): list of particle positions
        model_params: Params namedtuple, see pack_model_params

    Returns:
        L (float): log of the probability of the trajectory (sum of the probabilities of each transition)
    """
    N = len(S_arr)
    L = 0.
    X_tether_arr = calc_x_tether(S_arr, X_arr)
    for n in range(1, N):
        L += transition_log_probability(S_arr[n - 1], S_arr[n], X_arr[[n - 1]], X_arr[[n]], X_tether_arr[[n - 1]],
                                              model_params)[0]

    return L


def calc_x_tether(S_arr, X_arr):
    """
    Calculates array of X_tether (X^*) from arrays of S and X. X_tether always stays the same unless both S[n]=0 and
    S[n+1]=1, and then X_tether[n+1] = X[n].
    """
    N = len(S_arr)
    X_tether_arr = np.zeros([N, 2])
    X_tether_arr[0] = X_arr[0]
    for n in range(1, N):
        if (S_arr[n - 1] == 0) and (S_arr[n] != 0):
            X_tether_arr[n] = X_arr[n]
        else:
            X_tether_arr[n] = X_tether_arr[n - 1]
    return X_tether_arr


def generate_synthetic_trajectories(N_steps: int, N_particles: int, dt: float, T_stick: float, T_unstick: float,
                                    D: float, A: float, random_seed=None):
    """
    this is a utility function wrapping the "generate_model_trajectories" function
    Args:
        N_steps: Number of steps
        N_particles: Number of particles
        dt: delta t sampling interval
        T_stick: model parameter
        T_unstick: model parameter
        D: model parameter
        A: model parameter
        random_seed: random seed for trajectory generation (two identical seeds yield identical dataframes)
    Returns:
        pandas DataFrame containing the trajectories
    """
    init_S = None  # random
    model_params = pack_model_params(T_stick, T_unstick, D, A, dt)  # for convenience
    states_arr, X_arr, X_tether_arr = generate_trajectories(N_steps, N_particles, init_S, model_params,
                                                                  random_seed=random_seed)

    df = pd.DataFrame([])
    for i in range(N_particles):
        cur_df = pd.DataFrame({
            "particle": i,
            "frame": np.arange(N_steps, dtype=int),
            "t": np.arange(N_steps, dtype=int) * dt,
            "x": X_arr[i][:, 0],
            "y": X_arr[i][:, 1],
            "state": states_arr[i],
            "x_tether": X_tether_arr[i][:, 0],
            "y_tether": X_tether_arr[i][:, 1],
        })
        df = pd.concat([df, cur_df])

    return df, model_params
