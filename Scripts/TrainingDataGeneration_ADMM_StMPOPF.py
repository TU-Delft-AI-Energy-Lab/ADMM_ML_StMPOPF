"""
====================================================================
Training Data Generation for ML-Accelerated ADMM for Stochastic AC MP-OPF 
====================================================================

Author: Ali Rajaei  
Affiliation:  
- Department of Electrical Sustainable Energy, Delft University of Technology, the Netherlands  
Contact: a.rajaei@tudelft.nl  
Date: 2025

This implementation was developed as part of the research paper:

> Rajaei, Ali, Olayiwola Arowolo, and Jochen L. Cremer.  
> "Learning-Accelerated ADMM for Stochastic Power System Scheduling with Numerous Scenarios."  
> *IEEE Transactions on Sustainable Energy*, 2025.

License:
--------
MIT License (c) 2025 Ali Rajaei
"""


# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random



# %%
def generate_1yr_loaddata(networkdata, load_profile, lowD=0.95, highD=1.05, correlation=False, print_result=False):
    """
    Generates a synthetic one-year hourly load dataset based on the network structure
    and a given normalized load profile. Each demand is scaled by a random factor 
    within [lowD, highD] and the corresponding nominal demand.

    Parameters
    ----------
    networkdata : dict
        Dictionary containing network data (e.g., Bus, Demandset, D2B, Pd_nominal).
    load_profile : array-like
        Normalized load profile of shape (8760,) for one year.
    lowD : float, optional
        Lower bound for random scaling of demand (default is 0.95).
    highD : float, optional
        Upper bound for random scaling of demand (default is 1.05).
    correlation : bool, optional
        If True, introduces correlation in the noise model (not implemented yet).
    print_result : bool, optional
        If True, prints status messages.

    Returns
    -------
    X : ndarray of shape (n_bus, 8760)
        Hourly active load values at each bus for a one-year simulation.
    """

    Bus = networkdata['Bus']
    DemandSet = networkdata['Demandset']
    D2B = networkdata['D2B']
    Pd_nominal = networkdata['Pd_nominal']

    Nload = len(DemandSet)
    Nbus = len(Bus)
    NT = len(load_profile)

    # Generate random scaling factors for each load
    np.random.seed(1995)
    Xd = np.random.uniform(low=lowD, high=highD, size=(Nload, NT))

    # Initialize full load matrix
    X = np.zeros((Nbus, NT))

    # Map scaled load profile to corresponding buses
    for d_ind, d in enumerate(DemandSet):
        for _, b in D2B.select(d, '*'):
            b_idx = Bus.index(b)
            X[b_idx, :] = Xd[d_ind, :] * load_profile * Pd_nominal[d_ind]

    if correlation:
        kumaraswamy_montecarlo(a=1.6, b=2.8, c=correlation, 
                                    lower_bounds=np.repeat(lowD,Nload), upper_bounds=np.repeat(highD,Nload),num_samples=Nsamples).T

    if print_result:
        print(f'Generated load matrix X with shape ({Nbus}, {NT})')

    return X

# %%
def generate_scenarios_toyexample(Pd_array, NumSc=20, dev=0.01, seed=0, print_result=False):
    """
    Generates multiple demand scenarios from a base Pd_array using a multiplicative
    stochastic process.

    Note: This is a toy example for illustration. The scenario generation should be done with ARIMA models.

    Parameters
    ----------
    Pd_array : ndarray of shape (Nbus, T)
        Deterministic base load profile across buses and time steps.
    NumSc : int, optional
        Number of scenarios to generate (default is 20).
    dev : float, optional
        Standard deviation (or range) of noise added (default is 0.01).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    print_result : bool, optional
        If True, plots scenario profiles and standard deviation.

    Returns
    -------
    Pd_sc_array : ndarray of shape (NumSc, Nbus, T)
        Scenario-wise load data (Scenarios , Buses , Time).
    """

    T = Pd_array.shape[1]
    Nbus = Pd_array.shape[0]
    Pd_sc_array = np.zeros((NumSc, Nbus, T))

    # Copy initial time step across all scenarios
    Pd_sc_array[:, :, 0] = Pd_array[:, 0]

    # Initialize scaling factors
    pi = np.ones(NumSc)

    # Generate scenario variations over time using normal noise
    for t in range(1, T):
        np.random.seed(seed + t)
        pi *= np.random.normal(1, dev / 3, size=NumSc)  # Use Gaussian noise for temporal correlation
        for sc in range(NumSc):
            Pd_sc_array[sc, :, t] = pi[sc] * Pd_array[:, t]

    if print_result:
        # Plot total load over time for each scenario
        plt.figure(figsize=(10, 6))
        for sc in range(NumSc):
            plt.plot(Pd_sc_array[sc].sum(axis=0), label=f"Sc {sc}")
        plt.plot(Pd_sc_array.sum(axis=1).mean(axis=0), linestyle='--', linewidth=2,
                 color='black', label='Mean')
        plt.title('Scenario Total Load Over Time')
        plt.xlabel('Time')
        plt.ylabel('Total Load')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot standard deviation across scenarios at each time step
        plt.figure(figsize=(5, 3))
        plt.title('Standard Deviation of Total Load per Time Step')
        plt.plot(Pd_sc_array.sum(axis=1).std(axis=0))
        plt.xlabel('Time')
        plt.ylabel('Std Dev')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return Pd_sc_array
        

# %%
def generate_stochastic_load_1yr(networkdata, load_profile, 
                                 lowD=0.98, highD=1.02, 
                                 T=24, NumSc=20, dev=0.01, 
                                 print_result=False):
    """
    Generate stochastic 1-year load scenarios with temporal correlation for ADMM training.

    Parameters
    ----------
    networkdata : dict
        Power system data including buses, generator mapping, and nominal demand.
    load_profile : ndarray of shape (8760,)
        Normalized hourly load shape for one year.
    lowD : float, optional
        Minimum demand multiplier (default is 0.98).
    highD : float, optional
        Maximum demand multiplier (default is 1.02).
    T : int, optional
        Number of consecutive time steps in each training sample (default is 24).
    NumSc : int, optional
        Number of scenarios to generate per sample (default is 20).
    dev : float, optional
        Standard deviation of random noise for stochastic variation (default is 0.01).
    print_result : bool, optional
        If True, plots will be generated for inspection (default is False).

    Returns
    -------
    dict
        {   'X_sc': ndarray of shape (Nbus, NT, NumSc, T),
                Stochastic demand samples over time windows.
            'X_true': ndarray of shape (Nbus, 8760),
                Deterministic base load profile.}
    """

    Bus = networkdata['Bus']
    

    Nbus = len(Bus)
    NT = len(load_profile) - T  # number of time windows in the year

    # Generate deterministic load profile (shape: [Nbus, 8760])
    X_true = generate_1yr_loaddata(networkdata, load_profile, lowD=lowD, highD=lowD, print_result=print_result)

    # Initialize stochastic data tensor: (Buses, Time Windows, Scenarios, Time Steps)
    Xd_t_sc = np.zeros((Nbus, NT, NumSc, T))

    for t in tqdm(range(NT), desc="Generating stochastic scenarios"):
        # Slice a deterministic window of T hours
        base_window = X_true[:, t:t+T]
        
        # Generate NumSc stochastic variations of this window (Sc, Bus, T)
        x_d_sc = generate_scenarios_toyexample(Pd_array=base_window, NumSc=NumSc, dev=dev, seed=t, print_result=False) #replace with ARIMA version
        
        # Rearrange to (Bus, Sc, T) and store in time index
        Xd_t_sc[:, t, :, :] = x_d_sc.transpose((1, 0, 2))

    if print_result:
        print("Xd_t_sc shape:", Xd_t_sc.shape)

    return {
        'X_sc': Xd_t_sc,   # stochastic scenarios
        'X_true': X_true   # base deterministic year
    }



# %%
def generate_TrainingData_ADMM_ACOPF(
    networkdata,
    load_profile,
    T=24,
    lowD=0.95,
    highD=1.05,
    dev=0.02,
    NumSc=20,
    ADMM_iter=10,
    rho=0.1,
    lamda_std=0.5,
    exploration=False,
    explore_iter=0,
    exp_decay=0.1,
    Nsamples='all',
    print_result=False
):
    """
    Generates training data for Learning-Accelerated ADMM using rolling windows
    of stochastic load and solving ACOPF for each sample using the ADMM procedure.

    Parameters
    ----------
    networkdata : dict
        Power system data including Bus, G, Demandset, etc.
    load_profile : np.ndarray
        Yearly normalized demand vector (8760,).
    T : int
        Length of each sample window in hours.
    lowD, highD : float
        Range for base load scaling factors.
    dev : float
        Std deviation for scenario generation noise.
    NumSc : int
        Number of scenarios per sample.
    ADMM_iter : int
        Number of ADMM iterations.
    rho : float
        ADMM penalty parameter.
    lamda_std : float
        Std deviation for lambda initialization.
    exploration : bool
        If True, dual variable exploration is enabled.
    explore_iter : int
        Number of iterations to apply exploration.
    exp_decay : float
        Decay rate of exploratory updates.
    Nsamples : int or 'all'
        Number of training samples to extract.
    print_result : bool
        If True, enables debug visuals.

    Returns
    -------
    dict
        {
            'Xd':      [Nsamples, ADMM_iter, Sc, B, T],
            'Lambda':  [Nsamples, ADMM_iter, Sc, G, T],
            'PgX':     [Nsamples, ADMM_iter, Sc, G, T],
            'PgZ':     [Nsamples, ADMM_iter, Sc, G, T],
            'rk':      [Nsamples, ADMM_iter]
        }
    """

    Bus = networkdata['Bus']
    G = networkdata['G']
    Nbus = len(Bus)
    NG = len(G)

    if Nsamples == 'all':
        Nsamples = len(load_profile) - 2 * T
    else:
        print(f"{Nsamples} samples requested!")

    # Sample time indices randomly for rolling windows
    random.seed(1)
    indices = random.sample(range(T, len(load_profile) - T), Nsamples)

    if print_result:
        plt.figure(figsize=(3, 2))
        plt.hist(indices, bins=100)
        plt.title("Sampled Time Indices")
        plt.show()

    print(f"\nGenerating {Nsamples} training samples using {T}-hour rolling windows and {ADMM_iter} ADMM iterations.\n")

    # Generate stochastic demand data
    res = generate_stochastic_load_1yr(
        networkdata, load_profile, lowD=lowD, highD=highD, dev=dev, T=T, NumSc=NumSc, print_result=False
    )
    load_array = res['X_sc']  # shape: [B, NT, Sc, T]
    X_true = res['X_true']    # shape: [B, 8760]

    # Reorder to: [NT, Sc, B, T]
    Xd_all = load_array.transpose((1, 2, 0, 3))  # shape: [NT, Sc, B, T]

    print("Xd shape is:", Xd_all.shape)

    # Initialize outputs
    Xd = np.zeros((Nsamples, NumSc, Nbus, T))
    y = np.zeros((Nsamples, ADMM_iter, NumSc, NG, T))
    lambda_k = np.random.normal(0, lamda_std, (Nsamples, ADMM_iter, NumSc, NG))
    PgX_k = np.zeros((Nsamples, ADMM_iter, NG, T))
    rk = np.zeros((Nsamples, ADMM_iter))

    # Initialize models
    modelX = create_XUpdate_Stochastic_ACOPF(networkdata, NumSc=NumSc, nT=T, rho=rho, Ramp=True, Tlink=False)
    modelZ = create_ZUpdate_Stochastic_ACOPF(networkdata, nT=T, rho=rho, Ramp=True, Tlink=False)

    # Run ADMM for each training sample
    for i, sample in enumerate(tqdm(indices, desc="Running ADMM")):
        demand_sample = Xd_all[sample]  # [Sc, B, T]
        Xd[i] = demand_sample

        res = Solve_ADMM_Stochastic_ACOPF(
            data=networkdata,
            modelX=modelX,
            modelZ=modelZ,
            lambda_initial=lambda_k[i, 0, :, :],
            lambda_std=lamda_std,
            exploration=exploration,
            explore_iter=explore_iter,
            exp_decay=exp_decay,
            Ramp=True,
            DemandInstnace=demand_sample,
            k_iter=ADMM_iter,
            rho=rho
        )

        y[i] = res['PgZ_k']
        lambda_k[i] = res['lambda_k']
        PgX_k[i] = res['PgX_k']
        rk[i] = res['rk']

    print('\nTraining data generation complete!')

    # Broadcast data to match ML input shapes
    Xd_new = np.repeat(Xd[:, np.newaxis, :, :, :], ADMM_iter, axis=1)
    lambda_new = np.repeat(lambda_k[:, :, :, :, np.newaxis], T, axis=4)
    PgX_new = np.repeat(PgX_k[:, :, np.newaxis, :, :], NumSc, axis=2)

    return {
        'Xd': Xd_new,
        'Lambda': lambda_new,
        'PgX': PgX_new,
        'PgZ': y,
        'rk': rk
    }




# End