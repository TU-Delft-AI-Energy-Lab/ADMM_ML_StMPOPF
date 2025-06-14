"""
====================================================================
ML-Accelerated ADMM for Stochastic AC MP-OPF (PyTorch + Pyomo)
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

Description:
------------
This Python script integrates supervised machine learning into the ADMM-based stochastic AC MP-OPF framework.
It trains a recurrent neural network (RNN) to predict optimal generator schedules across time steps and scenarios,
thereby accelerating the ADMM procedure by providing high-quality feasible solutions, which can be either directly used or warm start an ADMM solver.

Key Features:
-------------
- Recurrent Neural Network (RNN) model trained on ADMM-generated scenarios  
- CVXPY-based restoration layer to ensure power balance and ramping feasibility after ML predictions  
- Integration of RNN predictions into the ADMM algorithm  
- Weighted loss function   
- Dataset of ADMM-based stochastic AC MP-OPF samples  


Disclaimer:
-----------
This is a research prototype intended for academic use. It prioritizes modularity and clarity over efficiency.  
Users are encouraged to extend or customize the code for their own power system configurations and forecasting tasks.

Dependencies:
-------------
- Python 3.9+
- PyTorch
- NumPy
- CVXPY
- Pyomo + Gurobi
- Matplotlib (optional, for plotting)

License:
--------
MIT License (c) 2025 Ali Rajaei
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer



# %%  Restoration layer

def create_Balance_Restoration(data, nT=24, Ramp=True):
    """
    Create a convex optimization problem for balancing power generation and demand, as well as ramping constraints.

    This function uses CVXPY to define and return a problem that restores generation-demand
    balance and ramp constraints with minimal deviation from a predicted generation schedule.

    Parameters:
    -----------
    data : dict
        Contains grid data including generator information and bus mapping.
    nT : int, optional
        Number of time steps (default is 24).
    Ramp : bool, optional
        Whether to include ramping constraints (default is True).

    Returns:
    --------
    dict
        A dictionary containing:
            - 'problem': the CVXPY optimization problem
            - 'Pd': parameter for demand
            - 'Pg': decision variable for generator outputs
            - 'Pg_pred': parameter for predicted generation
            - 'variables': list of CVXPY variables
            - 'parameters': list of CVXPY parameters
    """

    ngen = len(data['G'])

    # === Variables ===
    Pg = cp.Variable(shape=(ngen, nT), nonneg=True, name="Pg")

    # === Parameters ===
    Pd = cp.Parameter(shape=(nT), name='Pd')                       # Total demand at each time step
    Pg_pred = cp.Parameter(shape=(ngen, nT), name='Pg_pred')       # Predicted generation values


    # === Constraints ===
    constraints = []

    for t in range(nT):
        # Power balance constraint: total generation must equal demand
        constraints += [cp.sum(Pg[:, t]) == Pd[t]]

        # Ramping constraints
        if Ramp and t != nT - 1:
            ramp_up = data['Gen_data']['RampUp'].to_numpy()
            ramp_dn = data['Gen_data']['RampDn'].to_numpy()
            constraints += [Pg[:, t+1] - Pg[:, t] <= ramp_up]
            constraints += [Pg[:, t] - Pg[:, t+1] <= ramp_dn]

    # === Objective: minimize squared deviation from predicted generation ===
    objective = cp.Minimize(cp.sum_squares(Pg - Pg_pred))

    # === Problem definition ===
    problem = cp.Problem(objective, constraints)

    return {
        'problem': problem,
        'Pd': Pd,
        'Pg': Pg,
        'Pg_pred': Pg_pred,
        'variables': [Pg],
        'parameters': [Pd, Pg_pred]
    }




# %% Training data in correct shape

def TrainingData_ADMM_RNN_Tseq(networkdata, dataset, Nsamples='all', seed=1):
    """
    Prepare sequential training data for RNN from ADMM-based dataset.

    Parameters
    ----------
    networkdata : dict
        Dictionary containing network topology and parameters (e.g., buses, generators).
    dataset : dict or np.lib.npyio.NpzFile
        Dictionary or loaded .npz file containing simulation outputs: Xd, PgX, Lambda, PgZ.
        Shapes before transpose:
            - Xd      : [Samples, ADMM_iter, Sc, B, T]
            - PgX     : [Samples, ADMM_iter, Sc, G, T]
            - Lambda  : [Samples, ADMM_iter, Sc, G, T]
            - PgZ     : [Samples, ADMM_iter, Sc, G, T]
    Nsamples : int or 'all', optional
        Number of samples to use for training. If 'all', uses all samples.
    seed : int, optional
        Random seed for shuffling the dataset.

    Returns
    -------
    dict
        Dictionary containing:
            - 'input' : shape [Samples, ADMM_iter, Sc, T, B + 2*G]
            - 'output': shape [Samples, ADMM_iter, Sc, T, G]
    """

    Bus = networkdata['Bus']
    Gen = networkdata['G']
    Gen_data = networkdata['Gen_data']
    DemandSet = networkdata['Demandset']
    G2B = networkdata['G2B']
    Pd_array = networkdata['Pd_array']

    # Extract arrays from dataset (either dict or npz)
    Xd = dataset['Xd']
    PgX = dataset['PgX']
    Lambda = dataset['Lambda']
    PgZ = dataset['PgZ']

    T = Xd.shape[4]  # Time steps

    print(f"Reading dataset ...")
    print(f"Xd shape is [Samples, ADMM_iter, Sc, B, T]: {Xd.shape}")
    print(f"{Nsamples} samples requested!")

    if Nsamples == 'all':
        Nsamples = len(Xd)

    # === Transpose to match RNN sequential format ===
    Xd = Xd.transpose((0, 1, 2, 4, 3))
    PgX = PgX.transpose((0, 1, 2, 4, 3))
    Lambda = Lambda.transpose((0, 1, 2, 4, 3))
    PgZ = PgZ.transpose((0, 1, 2, 4, 3))

    # === Concatenate input features ===
    input = np.concatenate((Xd, PgX, Lambda), axis=4)
    output = PgZ

    print("input shape [Samples, ADMM_iter, Sc, T, B + 2*G]:", input.shape)
    print("output shape [Samples, ADMM_iter, Sc, T, G]:", output.shape)
    print("[Samples, ADMM_iter, Sc] will be merged later for RNN batching")

    # === Shuffle the dataset ===
    np.random.seed(seed)
    permutation_index = np.random.permutation(len(input))
    input = input[permutation_index]
    output = output[permutation_index]

    return {
        'input': input,
        'output': output
    }

# %% Test ADMM-ML vs lables

def Test_ADMMwithML(networkdata, TestDataSet, MLmodel, lamda_std=0.3, ADMM_iter=10, rho=0.1, seed=0):
    """
    Run ADMM with learned z-updates with ML model and compare with standard ADMM and central stochastic optimization (SO) results.

    Parameters
    ----------
    networkdata : dict
        Network structure, generation data, and bus data.

    TestDataSet : dict
        Dictionary containing:
            - Xd: demand instances [Samples, Sc, Bus, T]
            - Lambda: ADMM lambdas from offline training [Samples, ADMM_iter, Sc, G]
            - rk: residuals from offline ADMM
            - Time_X_k, Time_Z_k: time per iteration in offline ADMM
            - Time_SO: time for solving stochastic optimal power flow (ground truth)
            - Pg_SO: optimal power from SO [Samples, Sc, G, T]
            - PgX_ADMM: power X-updates [Samples, ADMM_iter, G]
            - PgZ_ADMM: power Z-updates [Samples, ADMM_iter, Sc, G, T]

    MLmodel : PyTorch model
        Trained model that predicts lambda values given the current state.

    lamda_std : float
        Standard deviation for randomly initialized lambda.

    ADMM_iter : int
        Number of ADMM iterations to perform.

    rho : float
        ADMM penalty parameter.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary with all test results including prediction errors, runtimes, lambda distributions.
    """

    Gen = networkdata['G']

    Xd = TestDataSet['Xd'][:10]  # [Samples, Sc, Bus, T]
    Lambda_offline = TestDataSet['Lambda']  # [Samples, ADMM_iter, Sc, G]
    rk_offline = TestDataSet['rk']
    Time_X_k_offline = TestDataSet['Time_X_k']
    Time_Z_k_offline = TestDataSet['Time_Z_k']
    Time_SO = TestDataSet['Time_SO']

    PgZ_truth = TestDataSet['Pg_SO']  # [Samples, Sc, G, T]
    PgX_ADMM = TestDataSet['PgX_ADMM'][:, :, :]  # [Samples, ADMM_iter, G]
    PgZ_ADMM = TestDataSet['PgZ_ADMM']  # [Samples, ADMM_iter, Sc, G, T]

    print("Test_ADMMwithML: Xd shape [Samples, Sc, Bus, T]:", Xd.shape)

    Nsamples = Xd.shape[0]
    NumSc = Xd.shape[1]
    T = Xd.shape[3]

    # Initialize outputs
    PgX_ML = np.zeros((Nsamples, ADMM_iter, len(Gen)))
    PgZ_ML = np.zeros((Nsamples, ADMM_iter, NumSc, len(Gen), T))
    Lambda = np.random.normal(0, lamda_std, size=(Nsamples, ADMM_iter, NumSc, len(Gen)))
    rk = np.zeros((Nsamples, ADMM_iter))
    Time_X_k = np.zeros((Nsamples, ADMM_iter))
    Time_Z_k = np.zeros((Nsamples, ADMM_iter))
    error = np.zeros((Nsamples, ADMM_iter))
    error0 = np.zeros((Nsamples, ADMM_iter))

    # Build reusable modelX
    modelX = create_XUpdate_Stochastic_ACOPF(networkdata, NumSc=NumSc, nT=T, rho=rho, Ramp=True, Tlink=False)

    for i in tqdm(range(Nsamples)):
        res = ADMM_Stochastic_ACOPF_ML(        #runs ADMM-ML model
            data=networkdata,
            MLmodel=MLmodel,
            modelX=modelX,
            All_Sc=True,
            lambda_initial=Lambda[i, 0, :, :],
            Ramp=True,
            DemandInstnace=Xd[i, :, :, :],
            k_iter=ADMM_iter,
            rho=rho,
            print_result=False
        )

        PgZ_ML[i] = res['PgZ_k']
        PgX_ML[i] = res['PgX_k'][:, :, 0]
        Lambda[i] = res['lambda_k']
        rk[i] = res['rk']
        Time_X_k[i] = res['Time_X_k']
        Time_Z_k[i] = res['Time_Z_k']

    # Compute prediction errors
    for i in range(Nsamples):
        for k in range(ADMM_iter):
            error0[i, k] = np.square(PgX_ML[i, k, :] - PgZ_truth[i, 0, :, 0]).mean()
            error[i, k] = np.square(PgZ_ML[i, k] - PgZ_truth[i]).mean()

    error_offline = np.zeros((Nsamples, ADMM_iter))
    error0_offline = np.zeros((Nsamples, ADMM_iter))
    for i in range(Nsamples):
        for k in range(ADMM_iter):
            error0_offline[i, k] = np.square(PgX_ADMM[i, k, :] - PgZ_truth[i, 0, :, 0]).mean()
            error_offline[i, k] = np.square(PgZ_ADMM[i, k] - PgZ_truth[i]).mean()

    # === Plot Accuracy Errors ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
    y_min = min(error_offline.mean(axis=0))
    y_max = 1.0

    ax1.set_title('ADMM Error vs Truth')
    ax1.plot(error_offline.mean(axis=0), label='error')
    ax1.plot(error0_offline.mean(axis=0), label='error0')
    ax1.set_yscale('log')
    ax1.set_ylim(y_min, y_max)
    ax1.legend()

    ax2.set_title('ML Error vs Truth')
    ax2.plot(error.mean(axis=0), label='error')
    ax2.plot(error0.mean(axis=0), label='error0')
    ax2.set_yscale('log')
    ax2.set_ylim(y_min, y_max)
    ax2.legend()
    plt.show()

    # === Plot Residuals ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
    ax1.plot(rk_offline[:, :ADMM_iter].mean(axis=0), label='rk_offline')
    ax1.plot(rk.mean(axis=0), label='rk_ML')
    ax1.set_yscale('log')
    ax1.set_title('rk (Test Data)')
    ax1.legend()

    ax2.plot(rk.mean(axis=0), label='rk_ML')
    ax2.set_yscale('log')
    ax2.set_title('rk ML Only')
    plt.show()

    # === Plot Lambda Distributions ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
    ax1.set_title('Lambda (ADMM Offline)')
    ax1.hist(Lambda_offline[:, :, :, 0].flatten(), bins=20)
    ax2.set_title('Lambda (ML Prediction)')
    ax2.hist(Lambda[:, :, :, 0].flatten(), bins=20)
    plt.show()

    return {
        'Xd': Xd,
        'Lambda_ML': Lambda,
        'PgX_ML': PgX_ML,
        'PgZ_ML': PgZ_ML,
        'error0_ML': error0,
        'error_ML': error,
        'error0_offline': error0_offline,
        'error_offline': error_offline,
        'PgZ_truth': PgZ_truth,
        'PgX_ADMM_offline': PgX_ADMM,
        'PgZ_ADMM_offline': PgZ_ADMM,
        'Lambda_offline': Lambda_offline,
        'rk_ML': rk,
        'rk_offline': rk_offline,
        'Time_X_k_ML': Time_X_k,
        'Time_Z_k_ML': Time_Z_k,
        'Time_X_k_offline': Time_X_k_offline,
        'Time_Z_k_offline': Time_Z_k_offline,
        'Time_SO': Time_SO
    }


# %%


# %%

def ADMM_Stochastic_ACOPF_ML(data, MLmodel,
                              modelX=None, k_iter=10, All_Sc=True,
                              rho=10, rho_decay=3e-1, rho_min=1.0, lambda_initial=None,
                              Ramp=True, DemandInstnace=None, Pg0=None, print_result=False):
    """
    ADMM solver with ML-based Z-update for Stochastic AC MP-OPF.

    Args:
        data: Network data dictionary.
        MLmodel: Trained PyTorch model to predict PgZ.
        modelX: Pyomo model for X-update.
        k_iter (int): Number of ADMM iterations.
        All_Sc (bool): Whether to update all scenarios in Z-update.
        rho (float): Initial ADMM penalty parameter.
        rho_decay (float): Decay rate for rho.
        rho_min (float): Minimum value for rho.
        lambda_initial (np.ndarray): Initial dual variables (Sc x G).
        Ramp (bool): If True, applies ramp constraints.
        DemandInstnace (np.ndarray): [Sc, Bus, T] array of demands.
        Pg0: previous step Pg.
        print_result (bool): If True, print and plot diagnostics.

    Returns:
        Dict containing:
            - PgZ: Final PgZ [Sc, G, T]
            - PgX: Final PgX [G]
            - PgX_k: PgX at each iteration [k_iter, G, T]
            - PgZ_k: PgZ at each iteration [k_iter, Sc, G, T]
            - lambda_k: Duals at each iteration [k_iter, Sc, G]
            - rk: Residual history
            - Time_X_k: Time per X-update
            - Time_Z_k: Time per Z-update
    """
    
    G = data['G']

    if DemandInstnace is not None:
        Pd_sc_array = DemandInstnace.copy()
        EndTime = Pd_sc_array.shape[2] - 1
        T = range(0, EndTime + 1)
        NumSc = Pd_sc_array.shape[0]
        Senarios = range(0, NumSc)
    else:
        raise ValueError("DemandInstnace must be provided.")

    k_iter_list = range(k_iter)
    residual = {'rk': [], 'sk': []}
    Time_X_k = np.array([])
    Time_Z_k = np.array([])

    PgX = np.zeros((len(G)))
    PgZ = np.zeros((NumSc, len(G), len(T)))
    PgX_k = np.zeros((k_iter, len(G), len(T)))
    PgZ_k = np.zeros((k_iter, NumSc, len(G), len(T)))

    if lambda_initial is None:
        lambda_s = np.random.normal(0, 0.1, size=(NumSc, len(G)))
    else:
        lambda_s = lambda_initial

    lambda_k = np.zeros((k_iter, NumSc, len(G)))

    if modelX is None:
        modelX = create_XUpdate_Stochastic_ACOPF(data, NumSc=NumSc, nT=T, rho=rho, Ramp=True, Tlink=False)

    result = Solve_Xupdate_ACOPF(data, modelX, Pd_sc_array, PgZ[:, :, 0], lambda_s)
    PgX = result['Pg'][:, 0]
    PgZ[:, :, :] = result['Pg']

    start_time = time.time()

    for k in range(k_iter):
        lambda_k[k] = lambda_s

        result = Solve_Xupdate_ACOPF(data, modelX, Pd_sc_array, PgZ[:, :, 0], lambda_s)
        PgX = result['Pg'][:, 0]
        PgXt = result['Pg']
        PgX_k[k] = result['Pg']
        Time_X_k = np.append(Time_X_k, result['time'])

        if All_Sc:
            # ML input is [batch,T,B+2G]
            # ML output is [batch,T,G], PgZ is [Sc,G,T]
            # Pd_sc_array is [Sc,Bus,T] -> [SC,T,B]
            # lambda_s is [Sc,G]  -> [Sc,T,G]
            # PgX is [G,T] -> [Sc,T,G]
            MLmodel.to(device)
            MLmodel.eval()

            Pd_input = Pd_sc_array.transpose((0, 2, 1))  # [Sc, T, B]
            lambda_input = np.zeros((NumSc, len(T), len(G)))
            lambda_input[:, 0, :] = lambda_s
            lambda_input[:, 1:, :] = lambda_s[:, np.newaxis, :]

            PgX_input = np.tile(PgXt.T[np.newaxis, :, :], (NumSc, 1, 1))  # [Sc, T, G]
            inputt = np.concatenate((Pd_input, PgX_input, lambda_input), axis=2)
            inputt = torch.Tensor(inputt).to(device)

            st = time.time()
            PgZ_pred = MLmodel(inputt)[0].detach()
            et = time.time()
            Time_Z_k = np.append(Time_Z_k, et - st)

            PgZ = PgZ_pred.cpu().numpy().transpose((0, 2, 1))  # [Sc, G, T]

            for g_ind, g in enumerate(G):
                for sc_ind, sc in enumerate(Senarios):
                    lambda_s[sc_ind, g_ind] += rho * (PgX[g_ind] - PgZ[sc_ind, g_ind, 0])

        PgZ_k[k] = PgZ
        rk_now = np.square(PgX - PgZ[:, :, 0]).mean()
        residual['rk'].append(rk_now)

        rho = max(rho - rho_decay, rho_min)

    end_time = time.time()
    if print_result:
        print("\n******* finished!!! ********")
        print(f"ADMM time: {end_time - start_time:.3f} sec")

        plt.figure(figsize=(5, 3))
        plt.plot(range(k_iter), residual['rk'])
        plt.title('Residuals', fontsize=15)
        plt.xlabel('Iteration k', fontsize=12)
        plt.yscale('log')
        plt.show()

        plt.figure(figsize=(5, 3))
        for sc in Senarios:
            plt.plot(PgZ[sc, 0, :], label=sc)
        plt.plot(PgZ[:, 0, :].mean(axis=0), linestyle='dashed', linewidth=2, color='black', label='mean')
        plt.title('G1 for scenarios')
        plt.show()

    return {
        'PgZ': PgZ,
        'PgX': PgX,
        'PgX_k': PgX_k,
        'PgZ_k': PgZ_k,
        'lambda_k': lambda_k,
        'rk': residual['rk'],
        'Time_X_k': Time_X_k,
        'Time_Z_k': Time_Z_k
    }


# %%


# %%
def create_XUpdate_Stochastic_ACOPF(data, nT=24, NumSc=10, rho=1, Ramp=True, Tlink=False):
    """
    Creates a Pyomo model for the X-update step in stochastic AC Optimal Power Flow (ACOPF)
    using the ADMM method.

    This model is intended for multiple demand scenarios and includes optional ramping constraints
    and temporal linking of generator outputs.

    Parameters:
    - data: dict
        A dictionary containing all required network data including buses, generators, lines, etc.
        It must also include 'vmin' and 'vmax' keys for voltage limits.
    - nT: int
        Number of time steps (default: 24).
    - NumSc: int
        Number of demand scenarios (default: 10).
    - rho: float
        ADMM penalty parameter (default: 1).
    - Ramp: bool
        Whether to include generator ramp rate constraints (default: True).
    - Tlink: bool
        Whether to link the first period's generation to a predefined value Pg0 (default: False).

    Returns:
    - model: Pyomo ConcreteModel
        A Pyomo model that can be solved using any compatible solver.

    Notes:
    - ML input is [batch,T,B+2G]
    - ML output is [batch,T,G], PgZ is [Sc,G,T]
    - Pd_sc_array is [Sc,Bus,T] -> [SC,T,B]
    - lambda_s is [Sc,G]  -> [Sc,T,G]
    - PgX is [G,T] -> [Sc,T,G]
    """
    
    #======  data
    Bus = data['Bus']
    branch = data['branch']
    Lines = data['Lines']
    Gen_data = data['Gen_data']
    G = data['G']
    G2B = data['G2B']
    vmin = data['vmin']
    vmax = data['vmax']

    Senarios = range(0, NumSc)
    T = range(nT)
    EndTime = nT - 1

    model = ConcreteModel(name='ACOPF')
    model.Pg = Var(G, T, bounds=(0, None))
    model.Qg = Var(G, T, bounds=(None, None))
    model.lambdaP = Param(Senarios, G, initialize=0, mutable=True)
    model.PgZ = Param(Senarios, G, initialize=0, mutable=True)
    model.Pd = Param(Bus, T, initialize=0, mutable=True)
    model.Qd = Param(Bus, T, initialize=0, mutable=True)
    if Tlink:
        model.Pg0 = Param(G, mutable=True)

    model.V2 = Var(Bus, T, bounds=(0, None), initialize=1)
    model.L2 = Var(Lines, T, bounds=(0, None))
    model.Pflow = Var(Lines, T, bounds=(None, None))
    model.Qflow = Var(Lines, T, bounds=(None, None))
    model.OF = Var(bounds=(0, None))

    def eqPbalance(model, b, t):
        return sum(model.Pg[g, t] for g, b_ in G2B.select('*', b)) - model.Pd[b, t] == \
               sum(model.Pflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*')) - \
               sum(model.Pflow[l, i, j, t] - branch.loc[(l, i, j)]['r'] * model.L2[l, i, j, t]
                   for l, i, j in Lines.select('*', '*', b))
    model.eqPbalance = Constraint(Bus, T, rule=eqPbalance)

    def eqQbalance(model, b, t):
        return sum(model.Qg[g, t] for g, b_ in G2B.select('*', b)) - model.Qd[b, t] == \
               sum(model.Qflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*')) - \
               sum(model.Qflow[l, i, j, t] - branch.loc[(l, i, j)]['x'] * model.L2[l, i, j, t]
                   for l, i, j in Lines.select('*', '*', b))
    model.eqQbalance = Constraint(Bus, T, rule=eqQbalance)

    def eqSij(model, l, i, j, t):
        return model.Pflow[l, i, j, t]**2 + model.Qflow[l, i, j, t]**2 <= model.V2[i, t] * model.L2[l, i, j, t]
    model.eqSij = Constraint(Lines, T, rule=eqSij)

    def eqSij_V(model, l, i, j, t):
        return model.V2[i, t] - model.V2[j, t] == -branch.loc[(l, i, j)]['z2'] * model.L2[l, i, j, t] + \
               2 * (branch.loc[(l, i, j)]['r'] * model.Pflow[l, i, j, t] +
                    branch.loc[(l, i, j)]['x'] * model.Qflow[l, i, j, t])
    model.eqSij_V = Constraint(Lines, T, rule=eqSij_V)

    def eqVmax(model, i, t): return model.V2[i, t] <= vmax ** 2
    model.eqVmax = Constraint(Bus, T, rule=eqVmax)

    def eqVmin(model, i, t): return vmin ** 2 <= model.V2[i, t]
    model.eqVmin = Constraint(Bus, T, rule=eqVmin)

    def eqPijmax(model, l, i, j, t):
        return model.Pflow[l, i, j, t]**2 + model.Qflow[l, i, j, t]**2 <= branch.loc[(l, i, j)]['limit'] ** 2
    model.eqPijmax = Constraint(Lines, T, rule=eqPijmax)

    def eqPijmin(model, l, i, j, t):
        return -branch.loc[(l, i, j)]['limit'] ** 2 <= model.Pflow[l, i, j, t]**2 + model.Qflow[l, i, j, t]**2
    model.eqPijmin = Constraint(Lines, T, rule=eqPijmin)

    def eqPgmax(model, g, t): return model.Pg[g, t] <= Gen_data.loc[g]['Pmax']
    model.eqPgmax = Constraint(G, T, rule=eqPgmax)

    def eqPgmin(model, g, t): return Gen_data.loc[g]['Pmin'] <= model.Pg[g, t]
    model.eqPgmin = Constraint(G, T, rule=eqPgmin)

    def eqQgmax(model, g, t): return model.Qg[g, t] <= Gen_data.loc[g]['Qmax']
    model.eqQgmax = Constraint(G, T, rule=eqQgmax)

    def eqQgmin(model, g, t): return Gen_data.loc[g]['Qmin'] <= model.Qg[g, t]
    model.eqQgmin = Constraint(G, T, rule=eqQgmin)

    if Ramp:
        def eqRU(model, g, t):
            if t != EndTime:
                return model.Pg[g, t + 1] - model.Pg[g, t] <= Gen_data.loc[g]['RampUp']
            else:
                return Constraint.Skip
        model.eqRU = Constraint(G, T, rule=eqRU)

        def eqRD(model, g, t):
            if t != EndTime:
                return model.Pg[g, t] - model.Pg[g, t + 1] <= Gen_data.loc[g]['RampDn']
            else:
                return Constraint.Skip
        model.eqRD = Constraint(G, T, rule=eqRD)

        if Tlink:
            def eqPg0up(model, g): return model.Pg[g, T[0]] - model.Pg0[g] <= Gen_data.loc[g]['RampUp']
            def eqPg0dn(model, g): return model.Pg0[g] - model.Pg[g, T[0]] <= Gen_data.loc[g]['RampDn']
            model.eqPg0up = Constraint(G, rule=eqPg0up)
            model.eqPg0dn = Constraint(G, rule=eqPg0dn)

    model.eqOF = Constraint(expr=model.OF == sum(
        Gen_data.loc[g]['b'] * model.Pg[g, t] +
        0.0001 * Gen_data.loc[g]['b'] * model.Qg[g, t] ** 2
        for g in G for t in T
    ) + sum(model.lambdaP[sc, g] * (model.Pg[g, 0] - model.PgZ[sc, g]) +
            0.5 * rho * (model.Pg[g, 0] - model.PgZ[sc, g]) ** 2
            for g in G for sc in Senarios))

    model.obj = Objective(expr=model.OF, sense=pyo.minimize)

    return model


# %%

def Solve_Xupdate_ACOPF(data, modelX, Pd_sc_array, PgZ, lambda_s):
    """
    Solves the X-update step of the stochastic ACOPF problem for the demand at t0.

    Parameters:
    - data: dict
        Dictionary containing network data including Bus, G, Pd_array, Qd_array, etc.
    - modelX: Pyomo ConcreteModel
        The ACOPF model generated by create_XUpdate_Stochastic_ACOPF.
    - Pd_sc_array: ndarray, shape (NumSc, Bus, T)
        Scenario-based real power demands.
    - PgZ: ndarray, shape (NumSc, G)
        Z-update generator values from ADMM.
    - lambda_s: ndarray, shape (NumSc, G)
        Dual variable estimates for each scenario and generator.

    Returns:
    - dict with:
        - 'Pg': ndarray, shape (G, T): Solved generator output over time.
        - 'time': float: Solver execution time in seconds.
    """
    Bus = data['Bus']
    G = data['G']
    Pd_array = data['Pd_array']
    Qd_array = data['Qd_array']

    nT = Pd_sc_array.shape[2]
    NumSc = Pd_sc_array.shape[0]

    solver = SolverFactory('gurobi')

    # Estimate reactive power using the same Q/P ratio from base data
    Qd_sc_array = Pd_sc_array.copy()
    for b_ind in range(len(Bus)):
        if Pd_array[b_ind, 0] != 0:
            Qd_sc_array[:, b_ind, :] = Pd_sc_array[:, b_ind, :] * (Qd_array[b_ind, 0] / Pd_array[b_ind, 0])

    # Set averaged demand values in model
    for b_ind, b in enumerate(Bus):
        for t in range(nT):
            modelX.Pd[b, t].value = Pd_sc_array[:, b_ind, t].mean()
            modelX.Qd[b, t].value = Qd_sc_array[:, b_ind, t].mean()

    # Set Z-update and lambda values in model
    for sc in range(NumSc):
        for g_ind, g in enumerate(G):
            modelX.PgZ[sc, g].value = PgZ[sc, g_ind]
            modelX.lambdaP[sc, g].value = lambda_s[sc, g_ind]

    # Solve the model
    results = solver.solve(modelX, tee=False)
    ex_time = results['Solver'][0]['Time']

    # Extract generator output
    Pgt_array = np.zeros((len(G), nT))
    for ind, g in enumerate(G):
        for t in range(nT):
            Pgt_array[ind, t] = modelX.Pg[g, t].value

    return {
        'Pg': Pgt_array,
        'time': ex_time
    }
    

# %%
def run_casestudy(data_x_bus, dataset, T=6, seed=1, epochs=100, trained_model=None):
    """
    Runs training and validation of an RNN model for ADMM-based OPF prediction.

    Parameters:
    - data_x_bus: dict
        Grid topology and generator/bus data.
    - dataset: np.ndarray or dict
        Training dataset (inputs and outputs), either in dictionary or .npz format.
    - T: int
        Sequence length for RNN input.
    - seed: int
        Random seed for reproducibility.
    - epochs: int
        Number of training epochs.
    - trained_model: str or None
        Path to a pre-trained model to resume training.

    Returns:
    - dict with:
        - 'training_losses': list of float
        - 'validation_losses': list of float
        - 'model': trained PyTorch model
    """
    Bus = data_x_bus['Bus']
    Gen = data_x_bus['G']

    data = TrainingData_ADMM_RNN_Tseq(data_x_bus, dataset, Nsamples='all')
    Full_x, Full_y = torch.Tensor(data['input']), torch.Tensor(data['output'])
    del data
    print('input imported and deleted!')

    # Split into training and validation sets
    train_size = int((Full_x.shape[0] // 4) * 3)
    val_size = int((Full_x.shape[0] // 4))

    train_x = Full_x[:train_size, :, :].reshape(-1, T, len(Bus) + 2 * len(Gen))
    val_x = Full_x[train_size:train_size + val_size, :, :].reshape(-1, T, len(Bus) + 2 * len(Gen))
    del Full_x

    train_y = Full_y[:train_size, :, :].reshape(-1, T, len(Gen))
    val_y = Full_y[train_size:train_size + val_size, :, :].reshape(-1, T, len(Gen))
    del Full_y

    print("train_x.shape:", train_x.shape)
    print("train_y.shape:", train_y.shape)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    num_samples = train_x.shape[0]
    seq_length = train_x.shape[1]
    input_dim = train_x.shape[2]
    output_dim = train_y.shape[2]

    del train_x, train_y
    del val_x, val_y

    # these hyper-parameters should be adjusted
    batch_size = 16
    num_layers = 3
    hidden_feat = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    torch.manual_seed(seed)
    rnn_model = RNN(input_dim, hidden_feat, output_dim, num_layers, batch_size)
    print(rnn_model)

    # Count model parameters
    tot_params = sum(p.numel() for p in rnn_model.parameters())
    print(f"Total number of parameters = {tot_params}")

    # Load pretrained model if given
    if trained_model is not None:
        rnn_model.load_state_dict(torch.load(trained_model))
        print('\n\nLoading trained model weights!\n\n')

    # Training setup
    start = time.time()
    del train_dataset, val_dataset

    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.0001, weight_decay=5e-5)
    criterion = torch.nn.MSELoss()

    training_losses = []
    validation_losses = []

    for epoch in range(epochs + 1):
        print('epoch:', epoch)
        train_loss = train_rnn(rnn_model, train_loader, optimizer)
        valid_loss = evaluate_rnn(rnn_model, val_loader)

        training_losses.append(train_loss)
        validation_losses.append(valid_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}')
            print(f'\tTrain Loss: {train_loss:.4f}')
            print(f'\t Val. Loss: {valid_loss:.4f}')

    end = time.time()

    # Plot loss
    plt.subplots(figsize=(5, 3))
    plt.plot(range(len(training_losses)), training_losses, 'r', label='Training loss', linestyle='solid')
    plt.plot(range(len(validation_losses)), validation_losses, 'g', label='Validation loss', linestyle='dashed')
    plt.legend()
    plt.title(f'RNN Training and Validation Loss for {system_size} Bus System', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.show()

    print('Training time is: %.2f seconds' % (end - start))

    return {
        'training_losses': np.array(training_losses),
        'validation_losses': np.array(validation_losses),
        'model': rnn_model
    }



# %%
class RNN(nn.Module):
    """
    A recurrent neural network model for generator power prediction (Pg).

    The model:
    - Takes time series inputs of demand and lambda values
    - Predicts generator power (Pg) using an RNN followed by a fully connected layer
    - Applies sigmoid activation and scales by Pmax
    - Applies a balancing layer to enforce nodal power balance

    Args:
    - input_dim: int, number of input features (Bus + 2 * Gen)
    - hidden_dim: int, size of RNN hidden layer
    - output_dim: int, number of generators (G)
    - num_layers: int, number of RNN layers
    - batch_size: int, batch size used during training
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.pmax = torch.Tensor(data_x_bus['Gen_data']['Pmax'].values).to(device)
        self.NT = 24
        self.NB = len(data_x_bus['Bus'])

    def forward(self, x):
        x = x.to(device)
        batch_size = x.size(0)
        seq_length = x.size(1)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        out = F.sigmoid(out).to(device)

        out = out * self.pmax

 
        Pg_pred = torch.transpose(out, 1, 2).to(device)  # [batch,G,T]
        xd = torch.transpose(x, 1, 2)[:,:self.NB,:].sum(axis=1).to(device)  # [batch,Bus,T]=>[batch,T]

        Pg_r, = layer_balance(xd, Pg_pred)  # [batch,G,T]

        Pg_r = torch.transpose(Pg_r, 1, 2)     # [batch,T,G]
        Pg_pred = torch.transpose(Pg_pred, 1, 2)  # [batch,T,G]

        return Pg_r, Pg_pred

# %%
def train_rnn(model, loader, optimizer, device=device):
    """
    Trains the RNN model for one epoch using the provided DataLoader.

    The loss includes:
    - MSE between predicted and target Pg values
    - A secondary term comparing the predicted output to the raw model output before balancing
    - A weighted loss at t=0 to prioritize early-step accuracy
    - An ADMM-based penalty term incorporating dual variables and consensus constraints

    Args:
    - model: the RNN model to train
    - loader: DataLoader object containing (input, target) pairs
    - optimizer: optimizer used for parameter updates
    - device: computation device (default: global `device` either cpu or cuda versions)

    Returns:
    - Average training loss over all batches
    """

    NG = len(data_x_bus['G'])  # Number of generators
    ND = system_size           # Number of buses

    model.to(device)
    model.train()
    epoch_loss = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)   # shape: [batch, T, D+PgX+Lambda]
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass through model
        outputs, Pg_pred = model(inputs)

        # Base loss: MSE between final balanced output and targets
        loss = criterion(outputs, targets) + 0.5 * criterion(outputs, Pg_pred)

        # Emphasize time step t=0 (initial step) accuracy
        loss_t0 = criterion(outputs[:, 0, :], targets[:, 0, :])

        # ADMM consensus loss (enforces agreement with PgX and dual lambda_s at t=0)
        PgX = inputs[:, 0, ND:ND+NG].detach()        # [batch, G]
        lambda_s = inputs[:, 0, ND+NG:].detach()     # [batch, G]
        rho = 0.1

        ADMM_loss = 1e-2 * lambda_s * (outputs[:, 0, :] - PgX) + \
                    1e-1 * (rho / 2) * torch.square(outputs[:, 0, :] - PgX)
        ADMM_loss = ADMM_loss.mean()

        # Final weighted loss composition (tuned heuristically)
        loss = loss + 10 * loss_t0 + 1 * ADMM_loss

        # Backpropagation and update
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


# %%
def evaluate_rnn(model, loader, device=device):
    """
    Evaluates the RNN model on the validation set.

    Args:
    - model: the trained RNN model
    - loader: DataLoader object containing (input, target) pairs for validation
    - device: computation device (default: global `device`)

    Returns:
    - Average validation loss over all batches
    """

    model.to(device)
    model.eval()  # Set model to evaluation mode

    epoch_loss = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)[0]  # Only need balanced output
            loss = criterion(outputs, targets)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)



# %%
def test_rnn(model, loader, device=device):
    """
    Runs the trained RNN model on test data and collects predictions and true outputs.

    Args:
    - model: trained RNN model
    - loader: DataLoader object containing test (input, target) batches
    - device: computation device (default: global `device`)

    Returns:
    - Average test loss
    - List of predicted generator outputs (per batch)
    - List of true generator outputs (per batch)
    """

    model.to(device)
    model.eval()  # Set model to evaluation mode

    epoch_loss = 0
    model_preds = []
    actual_y = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)[0]  # Use only balanced output
            loss = criterion(outputs, targets)

            model_preds.append(outputs.cpu())
            actual_y.append(targets.cpu())
            epoch_loss += loss.item()

    return epoch_loss / len(loader), model_preds, actual_y


# %% Run from here!


# Importing datasets and required functions
data_file = 'IEEE_118_bus_Data_PGLib_ACOPF.xlsx'
multiopf_dataset = '....npz'
system_size = 118

exec(open('ADMM_StMPOPF.py').read())


# Setting up GPU and checking RAM availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

try:
    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    print(f"Available RAM: {ram_gb:.1f} GB")
except:
    print("Unable to check RAM availability")

try:
    gpu_info = !nvidia-smi
    gpu_info = '\n'.join(gpu_info)
    print(gpu_info if 'failed' not in gpu_info else 'GPU not available')
except:
    print("Unable to check GPU info")

# %% [markdown]
# Create the restoration layer and read datasets

# %%
data_x_bus = read_data_ACOPF(File=data_file, print_data=False)
res = create_Balance_Restoration(data_x_bus, nT=24, Ramp=False)
print('Creating restoration layer...')
layer_balance = CvxpyLayer(res['problem'], parameters=res['parameters'], variables=res['variables'])
print('Restoration layer created.')

dataset = np.load(multiopf_dataset)



# %% Run this block to train the RNN model
criterion = torch.nn.MSELoss()

NumRuns = 1
epochs = 1

train_loss_array = np.zeros((NumRuns, epochs + 1))
val_loss_array = np.zeros((NumRuns, epochs + 1))
Models = []

for run in range(NumRuns):
    print(f"\n\n=== Run {run} ===\n")
    seed = run
    trained_model = None

    result = run_casestudy(data_x_bus, dataset, T=24, seed=seed, epochs=epochs, trained_model=trained_model)

    train_loss_array[run, :] = result['training_losses']
    val_loss_array[run, :] = result['validation_losses']
    Models.append(result['model'])
    trained_model = result['model']

# %% ML vs Test Data with Labels


sys = 118
Nsamples = 
rho_off = 
rho_ML = 
lam_off = 
lam_ML = 
NumSc = 
seed = 
ADMM_iter = 
nT = 

TestDataSet = np.load('TestDataset_....npz')

r = Test_ADMMwithML(data_x_bus, TestDataSet, MLmodel=trained_model, lamda_std=lam_ML, ADMM_iter=ADMM_iter, rho=rho_ML, seed=seed)

# End

