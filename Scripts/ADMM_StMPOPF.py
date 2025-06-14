"""
====================================================================
Stochastic ACOPF with Learning-Accelerated ADMM (Pyomo-Gurobi)
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
This Python script provides an open-source implementation of an ADMM-based 
decomposition algorithm for solving stochastic AC Optimal Power Flow (ACOPF) problems. 
It supports realistic power system scheduling under uncertainty with scenario-wise 
decomposition and temporal coupling.

Key Features:
-------------
- ACOPF modeling using Pyomo with Gurobi as the solver
- ADMM framework for scalable scenario decomposition
- X-update and Z-update subproblem structure with ramping and voltage constraints
- Support for exploration (randomized dual updates) during ADMM iterations, used for ML training data generation


Disclaimer:
-----------
This is a research prototype for academic purposes. It prioritizes clarity and modularity 
over computational efficiency. Contributions and extensions are welcome.

Usage Example at the end of the description

Dependencies:
-------------
- Python 3.9+
- Pyomo
- Gurobi
- NumPy
- Matplotlib (optional, for plotting)

License:
--------
MIT License (c) 2025 Ali Rajaei
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import itertools
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import networkx as nx
import time
from copy import deepcopy
import math


from pyomo.environ import *
import pyomo.environ as pyo


# %%
def read_data_ACOPF(File='IEEE_14_bus_Data.xlsx', print_data=False, DemFactor=1.0, LineLimit=1.0):
    """
    Reads and processes input data for stochastic ACOPF problems from Excel file.
    
    Parameters:
        File (str): Path to the Excel file containing input data.
        print_data (bool): If True, display loaded data and plot demand profile.
        DemFactor (float): Scaling factor for demand.
        LineLimit (float): Scaling factor for branch line limits.
    
    Returns:
        data (dict): A dictionary containing network, generator, and load data formatted for optimization.
    """
    Sbase = 100
    EndTime = 23
    T = range(0, EndTime + 1)
    data = {}

    # Load bus data
    Bus = pd.read_excel(File, sheet_name='Bus', index_col=0, usecols='A')
    Bus = list(Bus.index)

    # Load and process branch data
    branch = pd.read_excel(File, sheet_name='Branch', skiprows=1, index_col=[0, 1, 2], usecols='A:G')
    for l, i, j in branch.index:
        branch.loc[(l, j, i)] = branch.loc[(l, i, j)]  # Make branches bi-directional

    branch['limit'] = (LineLimit * branch['limit']) / Sbase
    branch['z2'] = branch['r']**2 + branch['x']**2
    branch['g_ij'] = branch['r'] / branch['z2']
    branch['b_ij'] = branch['x'] / branch['z2']
    branch['z'] = np.sqrt(branch['z2'])
    branch['th_ij'] = np.arctan(branch['x'] / branch['r'])

    Lines = gp.tuplelist(list(branch.index))  # Gurobi tuplelist for network connectivity

    # Map lines to buses
    L2B = gp.tuplelist([(l, i) for l, i, j in Lines])

    # Load demand set
    Pdemand = pd.read_excel(File, sheet_name='DemandSet', skiprows=1, index_col=2, usecols='A:E')
    Pdemand.drop(columns=['Unnamed: 0', 'Unnamed: 1'], inplace=True)
    Pdemand.rename(columns={1: 'Pd', 2: 'Qd'}, inplace=True)
    Pdemand.fillna(0, inplace=True)
    Pdemand['Pd'] = (DemFactor * Pdemand['Pd']) / Sbase
    Qdemand = Pdemand.copy()
    Qdemand['Qd'] = (DemFactor * Qdemand['Qd']) / Sbase

    Pd_nominal = Pdemand['Pd'].to_numpy()
    Qd_nominal = Qdemand['Qd'].to_numpy()

    # Scale loads over time using load profile
    LoadProfile = pd.read_excel(File, sheet_name='LoadProfile', index_col=0, usecols='A:B')
    for t in LoadProfile.index:
        Pdemand[t] = Pdemand['Pd'] * LoadProfile.loc[t, 'Pdt']
        Qdemand[t] = Qdemand['Qd'] * LoadProfile.loc[t, 'Pdt']

    DemandSet = list(Pdemand.index)
    D2B_df = pd.read_excel(File, sheet_name='D2B', index_col=[1, 2])
    D2B = gp.tuplelist(list(D2B_df.index))

    # Build demand array for all buses over time
    Pd_array = np.zeros((len(Bus), len(T)))
    Qd_array = np.zeros((len(Bus), len(T)))
    for d in DemandSet:
        for t in T:
            for dx, b in D2B.select(d, '*'):
                Pd_array[Bus.index(b), t] += Pdemand.loc[d, t]
                Qd_array[Bus.index(b), t] += Qdemand.loc[d, t]

    # Load generator data
    Gen_data = pd.read_excel(File, sheet_name='Gen', skiprows=2, index_col=0, usecols='A:K')
    Gen_data['Pmax'] = Gen_data['Pmax'] / Sbase
    Gen_data['Pmin'] = Gen_data['Pmin'] / Sbase
    Gen_data['RampDn'] = Gen_data['RampDn'] / Sbase
    Gen_data['RampUp'] = Gen_data['RampUp'] / Sbase
    Gen_data['a'] = Gen_data['a'] * Sbase  # Scale cost coefficients

    G = list(Gen_data.index)  # Generator set

    G2B_df = pd.read_excel(File, sheet_name='G2B', index_col=[1, 2])
    G2B = gp.tuplelist(list(G2B_df.index))

    # Optional print and plot
    if print_data:
        display(Pdemand)
        display(branch)
        display(Gen_data)
        plt.plot(T, Pdemand.sum(axis=0)[1:])
        plt.xlabel("Time")
        plt.ylabel("Total Demand (pu)")
        plt.title("Total Demand Over Time")
        plt.grid(True)
        plt.show()

    # Pack all data into dictionary
    data['Sbase'] = Sbase
    data['Bus'] = Bus
    data['branch'] = branch
    data['Lines'] = Lines
    data['L2B'] = L2B
    data['Pdemand'] = Pdemand
    data['Demandset'] = DemandSet
    data['D2B'] = D2B
    data['Gen_data'] = Gen_data
    data['G'] = G
    data['G2B'] = G2B
    data['EndTime'] = EndTime
    data['T'] = T
    data['Pd_array'] = Pd_array
    data['Qd_array'] = Qd_array
    data['LoadProfile'] = LoadProfile
    data['Pd_nominal'] = Pd_nominal
    data['Qd_nominal'] = Qd_nominal
    data['vmin'] = 0.9
    data['vmax'] = 1.1

    return data






# %% Non-linear ACOPF 

def create_MultiTime_ACOPF_Cos(data, nT=24, Ramp=True, Tlink=False):
    """
    Creates a Pyomo model for multi-time non-linear AC Optimal Power Flow (ACOPF).
    
    Parameters:
        data (dict): Input data dictionary containing buses, generators, branches, and load.
        nT (int): Number of time steps (default is 24).
        Ramp (bool): Whether to include ramp constraints.
        Tlink (bool): Whether to link ramp constraints to time 0 (initial state).
    
    Returns:
        model (ConcreteModel): Pyomo optimization model for ACOPF.
    """
    
    # Extract data
    Bus        = data['Bus']
    branch     = data['branch']
    Lines      = data['Lines']
    Gen_data   = data['Gen_data']
    G          = data['G']
    G2B        = data['G2B']
    vmin       = data['vmin']
    vmax       = data['vmax']

    EndTime = nT - 1
    T = range(0, EndTime + 1)

    # Initialize Pyomo model
    model = ConcreteModel(name='ACOPF')

    # Define variables
    model.Pg     = Var(G, T, bounds=(0, None))
    model.Qg     = Var(G, T, bounds=(None, None))
    model.Vol    = Var(Bus, T, bounds=(0, None))
    model.theta  = Var(Bus, T, bounds=(-math.pi/3, math.pi/3), initialize=0)
    model.Pflow  = Var(Lines, T, bounds=(None, None))
    model.Qflow  = Var(Lines, T, bounds=(None, None))
    model.Rup    = Var(G, T, bounds=(0, None))
    model.Rdn    = Var(G, T, bounds=(0, None))
    model.OF     = Var(bounds=(0, None))

    model.Pd = Param(Bus, T, initialize=0, mutable=True)
    model.Qd = Param(Bus, T, initialize=0, mutable=True)

    if Tlink:
        model.Pg0 = Param(G, mutable=True)

    vmin = 0.9
    vmax = 1.1

    # Power balance constraints
    def eqPbalance(model, b, t):
        return sum(model.Pg[g, t] for g, bb in G2B.select('*', b)) - model.Pd[b, t] == \
               sum(model.Pflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*'))
    model.eqPbalance = Constraint(Bus, T, rule=eqPbalance)

    def eqQbalance(model, b, t):
        return sum(model.Qg[g, t] for g, bb in G2B.select('*', b)) - model.Qd[b, t] == \
               sum(model.Qflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*'))
    model.eqQbalance = Constraint(Bus, T, rule=eqQbalance)

    # Flow symmetry
    def eqflow2(model, l, i, j, t):
        return model.Pflow[l, i, j, t] + model.Pflow[l, j, i, t] >= 0
    model.eqflow2 = Constraint(Lines, T, rule=eqflow2)

    # AC Power flow equations (Soroudi et al.)
    def eqPij(model, l, i, j, t):
        z_inv = 1 / branch.loc[(l, i, j)]['z']
        th = branch.loc[(l, i, j)]['th_ij']
        return model.Pflow[l, i, j, t] == \
            model.Vol[i, t] ** 2 * z_inv * cos(th) - \
            model.Vol[i, t] * model.Vol[j, t] * z_inv * cos(model.theta[i, t] - model.theta[j, t] + th)
    model.eqPij = Constraint(Lines, T, rule=eqPij)

    def eqQij(model, l, i, j, t):
        z_inv = 1 / branch.loc[(l, i, j)]['z']
        th = branch.loc[(l, i, j)]['th_ij']
        return model.Qflow[l, i, j, t] == \
            model.Vol[i, t] ** 2 * z_inv * sin(th) - \
            model.Vol[i, t] * model.Vol[j, t] * z_inv * sin(model.theta[i, t] - model.theta[j, t] + th)
    model.eqQij = Constraint(Lines, T, rule=eqQij)

    # Voltage limits
    model.eqVmax = Constraint(Bus, T, rule=lambda m, i, t: m.Vol[i, t] <= vmax)
    model.eqVmin = Constraint(Bus, T, rule=lambda m, i, t: m.Vol[i, t] >= vmin)

    # Flow limits
    model.eqPijmax = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2 <= branch.loc[(l, i, j)]['limit']**2)

    model.eqPijmin = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        -branch.loc[(l, i, j)]['limit']**2 <= m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2)

    # Generator capacity limits
    model.eqPgmax = Constraint(G, T, rule=lambda m, g, t: m.Pg[g, t] <= Gen_data.loc[g]['Pmax'])
    model.eqPgmin = Constraint(G, T, rule=lambda m, g, t: m.Pg[g, t] >= Gen_data.loc[g]['Pmin'])
    model.eqQgmax = Constraint(G, T, rule=lambda m, g, t: m.Qg[g, t] <= Gen_data.loc[g]['Qmax'])
    model.eqQgmin = Constraint(G, T, rule=lambda m, g, t: m.Qg[g, t] >= Gen_data.loc[g]['Qmin'])

    # Slack bus angle
    model.eqtheta = Constraint(T, rule=lambda m, t: m.theta[Bus[0], t] == 0)

    # Ramp constraints
    if Ramp:
        def eqRU(model, g, t):
            if t != EndTime:
                return model.Pg[g, t + 1] - model.Pg[g, t] <= Gen_data.loc[g]['RampUp']
            return Constraint.Skip
        model.eqRU = Constraint(G, T, rule=eqRU)

        def eqRD(model, g, t):
            if t != EndTime:
                return model.Pg[g, t] - model.Pg[g, t + 1] <= Gen_data.loc[g]['RampDn']
            return Constraint.Skip
        model.eqRD = Constraint(G, T, rule=eqRD)

        if Tlink:
            model.eqPg0up = Constraint(G, rule=lambda m, g: m.Pg[g, T[0]] - m.Pg0[g] <= Gen_data.loc[g]['RampUp'])
            model.eqPg0dn = Constraint(G, rule=lambda m, g: m.Pg0[g] - m.Pg[g, T[0]] <= Gen_data.loc[g]['RampDn'])

    # Objective: generation cost + small weight on reactive generation squared to reduce over Q production
    model.eqOF = Constraint(expr=model.OF == sum(
        Gen_data.loc[g]['b'] * model.Pg[g, t] +
        0.0001 * Gen_data.loc[g]['b'] * model.Qg[g, t] ** 2 
        for g in G for t in T
    ))

    model.obj = Objective(expr=model.OF, sense=pyo.minimize)

    # Define solver (you can optionally solve it outside)
    solver = SolverFactory('ipopt')

    return model



# %% SOCP-based ACOPF
def create_MultiTime_ACOPF_SOCP(data, nT=24, Ramp=True, Tlink=False):
    """
    Creates a Pyomo model for multi-time with SOCP relaxation
    of the AC ACOPF problem over a time horizon.

    Parameters:
        data (dict): Input dataset including bus, generator, branch, demand, etc.
        nT (int): Number of time steps (default: 24).
        Ramp (bool): Whether to apply ramp rate constraints.
        Tlink (bool): If True, applies ramping limits to t=0 from an initial value Pg0.

    Returns:
        model (ConcreteModel): Pyomo optimization model with SOCP relaxation of ACOPF.
    """

    # ==== Extract data
    Bus       = data['Bus']
    branch    = data['branch']
    Lines     = data['Lines']
    Gen_data  = data['Gen_data']
    G         = data['G']
    G2B       = data['G2B']
    vmin      = data['vmin']
    vmax      = data['vmax']

    EndTime = nT - 1
    T = range(0, EndTime + 1)

    # ==== Define model
    model = ConcreteModel(name='ACOPF_SOCP')

    # Variables
    model.Pg    = Var(G, T, bounds=(0, None))
    model.Qg    = Var(G, T, bounds=(None, None))
    model.Pd    = Param(Bus, T, initialize=0, mutable=True)
    model.Qd    = Param(Bus, T, initialize=0, mutable=True)
    if Tlink:
        model.Pg0 = Param(G, mutable=True)

    model.V2    = Var(Bus, T, bounds=(0, None), initialize=1)
    model.L2    = Var(Lines, T, bounds=(0, None))
    model.Pflow = Var(Lines, T, bounds=(None, None))
    model.Qflow = Var(Lines, T, bounds=(None, None))
    model.Rup   = Var(G, T, bounds=(0, None))
    model.Rdn   = Var(G, T, bounds=(0, None))
    model.OF    = Var(bounds=(0, None))

    # Power balance equations
    def eqPbalance(model, b, t):
        return (
            sum(model.Pg[g, t] for g, bb in G2B.select('*', b)) - model.Pd[b, t]
            == sum(model.Pflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*'))
             - sum(model.Pflow[l, i, j, t] - branch.loc[(l, i, j)]['r'] * model.L2[l, i, j, t]
                   for l, i, j in Lines.select('*', '*', b))
        )
    model.eqPbalance = Constraint(Bus, T, rule=eqPbalance)

    def eqQbalance(model, b, t):
        return (
            sum(model.Qg[g, t] for g, bb in G2B.select('*', b)) - model.Qd[b, t]
            == sum(model.Qflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*'))
             - sum(model.Qflow[l, i, j, t] - branch.loc[(l, i, j)]['x'] * model.L2[l, i, j, t]
                   for l, i, j in Lines.select('*', '*', b))
        )
    model.eqQbalance = Constraint(Bus, T, rule=eqQbalance)

    # SOCP relaxation constraints
    def eqSij(model, l, i, j, t):
        return model.Pflow[l, i, j, t]**2 + model.Qflow[l, i, j, t]**2 <= model.V2[i, t] * model.L2[l, i, j, t]
    model.eqSij = Constraint(Lines, T, rule=eqSij)

    def eqSij_V(model, l, i, j, t):
        return model.V2[i, t] - model.V2[j, t] == (
            -branch.loc[(l, i, j)]['z2'] * model.L2[l, i, j, t]
            + 2 * (branch.loc[(l, i, j)]['r'] * model.Pflow[l, i, j, t]
                 + branch.loc[(l, i, j)]['x'] * model.Qflow[l, i, j, t])
        )
    model.eqSij_V = Constraint(Lines, T, rule=eqSij_V)

    # Voltage bounds
    model.eqVmax = Constraint(Bus, T, rule=lambda m, i, t: m.V2[i, t] <= vmax ** 2)
    model.eqVmin = Constraint(Bus, T, rule=lambda m, i, t: m.V2[i, t] >= vmin ** 2)

    # Line flow limits
    model.eqPijmax = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2 <= branch.loc[(l, i, j)]['limit']**2)

    model.eqPijmin = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        -branch.loc[(l, i, j)]['limit']**2 <= m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2)

    # Generator limits
    model.eqPgmax = Constraint(G, T, rule=lambda m, g, t: m.Pg[g, t] <= Gen_data.loc[g]['Pmax'])
    model.eqPgmin = Constraint(G, T, rule=lambda m, g, t: m.Pg[g, t] >= Gen_data.loc[g]['Pmin'])
    model.eqQgmax = Constraint(G, T, rule=lambda m, g, t: m.Qg[g, t] <= Gen_data.loc[g]['Qmax'])
    model.eqQgmin = Constraint(G, T, rule=lambda m, g, t: m.Qg[g, t] >= Gen_data.loc[g]['Qmin'])

    # Ramping constraints
    if Ramp:
        def eqRU(model, g, t):
            if t != EndTime:
                return model.Pg[g, t + 1] - model.Pg[g, t] <= Gen_data.loc[g]['RampUp']
            return Constraint.Skip
        model.eqRU = Constraint(G, T, rule=eqRU)

        def eqRD(model, g, t):
            if t != EndTime:
                return model.Pg[g, t] - model.Pg[g, t + 1] <= Gen_data.loc[g]['RampDn']
            return Constraint.Skip
        model.eqRD = Constraint(G, T, rule=eqRD)

        if Tlink:
            model.eqPg0up = Constraint(G, rule=lambda m, g: m.Pg[g, T[0]] - m.Pg0[g] <= Gen_data.loc[g]['RampUp'])
            model.eqPg0dn = Constraint(G, rule=lambda m, g: m.Pg0[g] - m.Pg[g, T[0]] <= Gen_data.loc[g]['RampDn'])

    # Objective function: generation cost + small penalty on reactive power
    model.eqOF = Constraint(expr=model.OF >= sum(
        Gen_data.loc[g]['b'] * model.Pg[g, t] +
        0.0001 * Gen_data.loc[g]['b'] * model.Qg[g, t] ** 2
        for g in G for t in T
    ))
    model.obj = Objective(expr=model.OF, sense=pyo.minimize)

    return model



# %%  Stochastic ACOPF - SOCP

def create_Stochastic_ACOPF(data, nT=24, NumSc=10, Ramp=True, Tlink=False):
    """
    Creates a Pyomo model for stochastic AC Optimal Power Flow (ACOPF) with SOCP relaxation,
    considering multiple demand scenarios.

    Parameters:
        data (dict): Dataset with all required sets and parameters.
        nT (int): Number of time steps.
        NumSc (int): Number of demand scenarios.
        Ramp (bool): Whether to include ramping constraints.
        Tlink (bool): If True, adds ramping constraints to initial Pg0.

    Returns:
        model (ConcreteModel): Pyomo model representing the stochastic ACOPF problem.
    """

    # ==== Extract data
    Bus        = data['Bus']
    branch     = data['branch']
    Lines      = data['Lines']
    Gen_data   = data['Gen_data']
    G          = data['G']
    G2B        = data['G2B']
    vmin       = data['vmin']
    vmax       = data['vmax']

    Senarios = range(NumSc)
    T = range(nT)
    EndTime = nT - 1

    # ==== Model Definition
    model = ConcreteModel(name='Stochastic_ACOPF')

    # Variables
    model.Pg      = Var(G, T, Senarios, bounds=(0, None))
    model.Qg      = Var(G, T, Senarios, bounds=(None, None))
    model.Pg_t0   = Var(G, bounds=(0, None))  # Shared across scenarios
    model.V2      = Var(Bus, T, Senarios, bounds=(0, None), initialize=1)
    model.L2      = Var(Lines, T, Senarios, bounds=(0, None))
    model.Pflow   = Var(Lines, T, Senarios, bounds=(None, None))
    model.Qflow   = Var(Lines, T, Senarios, bounds=(None, None))
    model.OF      = Var(bounds=(0, None))

    # Parameters
    model.Pd = Param(Senarios, Bus, T, initialize=0, mutable=True)
    model.Qd = Param(Senarios, Bus, T, initialize=0, mutable=True)
    if Tlink:
        model.Pg0 = Param(G, mutable=True)

    # Power balance
    def eqPbalance(model, b, t, sc):
        return (
            sum(model.Pg[g, t, sc] for g, bb in G2B.select('*', b)) - model.Pd[sc, b, t]
            == sum(model.Pflow[l, i, j, t, sc] for l, i, j in Lines.select('*', b, '*'))
             - sum(model.Pflow[l, i, j, t, sc] - branch.loc[(l, i, j)]['r'] * model.L2[l, i, j, t, sc]
                   for l, i, j in Lines.select('*', '*', b))
        )
    model.eqPbalance = Constraint(Bus, T, Senarios, rule=eqPbalance)

    def eqQbalance(model, b, t, sc):
        return (
            sum(model.Qg[g, t, sc] for g, bb in G2B.select('*', b)) - model.Qd[sc, b, t]
            == sum(model.Qflow[l, i, j, t, sc] for l, i, j in Lines.select('*', b, '*'))
             - sum(model.Qflow[l, i, j, t, sc] - branch.loc[(l, i, j)]['x'] * model.L2[l, i, j, t, sc]
                   for l, i, j in Lines.select('*', '*', b))
        )
    model.eqQbalance = Constraint(Bus, T, Senarios, rule=eqQbalance)

    # SOCP constraints
    def eqSij(model, l, i, j, t, sc):
        return model.Pflow[l, i, j, t, sc]**2 + model.Qflow[l, i, j, t, sc]**2 <= model.V2[i, t, sc] * model.L2[l, i, j, t, sc]
    model.eqSij = Constraint(Lines, T, Senarios, rule=eqSij)

    def eqSij_V(model, l, i, j, t, sc):
        return model.V2[i, t, sc] - model.V2[j, t, sc] == (
            -branch.loc[(l, i, j)]['z2'] * model.L2[l, i, j, t, sc]
            + 2 * (branch.loc[(l, i, j)]['r'] * model.Pflow[l, i, j, t, sc]
                 + branch.loc[(l, i, j)]['x'] * model.Qflow[l, i, j, t, sc])
        )
    model.eqSij_V = Constraint(Lines, T, Senarios, rule=eqSij_V)

    # Voltage limits
    model.eqVmax = Constraint(Bus, T, Senarios, rule=lambda m, i, t, sc: m.V2[i, t, sc] <= vmax ** 2)
    model.eqVmin = Constraint(Bus, T, Senarios, rule=lambda m, i, t, sc: m.V2[i, t, sc] >= vmin ** 2)

    # Line flow limits
    model.eqPijmax = Constraint(Lines, T, Senarios, rule=lambda m, l, i, j, t, sc:
        m.Pflow[l, i, j, t, sc]**2 + m.Qflow[l, i, j, t, sc]**2 <= branch.loc[(l, i, j)]['limit']**2)

    model.eqPijmin = Constraint(Lines, T, Senarios, rule=lambda m, l, i, j, t, sc:
        -branch.loc[(l, i, j)]['limit']**2 <= m.Pflow[l, i, j, t, sc]**2 + m.Qflow[l, i, j, t, sc]**2)

    # Generator limits
    model.eqPgmax = Constraint(G, T, Senarios, rule=lambda m, g, t, sc: m.Pg[g, t, sc] <= Gen_data.loc[g]['Pmax'])
    model.eqPgmin = Constraint(G, T, Senarios, rule=lambda m, g, t, sc: m.Pg[g, t, sc] >= Gen_data.loc[g]['Pmin'])
    model.eqQgmax = Constraint(G, T, Senarios, rule=lambda m, g, t, sc: m.Qg[g, t, sc] <= Gen_data.loc[g]['Qmax'])
    model.eqQgmin = Constraint(G, T, Senarios, rule=lambda m, g, t, sc: m.Qg[g, t, sc] >= Gen_data.loc[g]['Qmin'])

    # Synchronize Pg across scenarios at t=0
    model.eqPgt0 = Constraint(G, Senarios, rule=lambda m, g, sc: m.Pg[g, 0, sc] == m.Pg_t0[g])

    # Ramping constraints
    if Ramp:
        def eqRU(model, g, t, sc):
            if t != EndTime:
                return model.Pg[g, t + 1, sc] - model.Pg[g, t, sc] <= Gen_data.loc[g]['RampUp']
            return Constraint.Skip
        model.eqRU = Constraint(G, T, Senarios, rule=eqRU)

        def eqRD(model, g, t, sc):
            if t != EndTime:
                return model.Pg[g, t, sc] - model.Pg[g, t + 1, sc] <= Gen_data.loc[g]['RampDn']
            return Constraint.Skip
        model.eqRD = Constraint(G, T, Senarios, rule=eqRD)

        if Tlink:
            model.eqPg0up = Constraint(G, rule=lambda m, g: m.Pg_t0[g] - m.Pg0[g] <= Gen_data.loc[g]['RampUp'])
            model.eqPg0dn = Constraint(G, rule=lambda m, g: m.Pg0[g] - m.Pg_t0[g] <= Gen_data.loc[g]['RampDn'])

    # Objective: cost with penalty on reactive power
    model.eqOF = Constraint(expr=model.OF >= sum(
        Gen_data.loc[g]['b'] * model.Pg[g, t, sc] +
        0.0001 * Gen_data.loc[g]['b'] * model.Qg[g, t, sc] ** 2
        for g in G for t in T for sc in Senarios
    ))
    model.obj = Objective(expr=model.OF, sense=pyo.minimize)

    return model
    

# %% toy example of how to use the Stochastic_ACOPF
# === Initialization ===
# sys    = 14
# nT     = 24
# NumSc  = 10
# T      = range(nT)
# Senarios = range(NumSc)

# # Read input data
# data   = read_data_ACOPF(File=f'IEEE_{sys}_bus_Data_PGLib_ACOPF.xlsx', DemFactor=1.0, print_data=False)
# Bus    = data['Bus']
# G      = data['G']
# Pd     = data['Pd_array']
# Qd     = data['Qd_array']

# # === Create toy stochastic demand scenarios ===
# np.random.seed(2023)
# pi = np.random.uniform(0.9, 1.1, size=NumSc)

# Pd_sc = np.zeros((NumSc, Pd.shape[0], Pd.shape[1]))
# Qd_sc = np.zeros((NumSc, Qd.shape[0], Qd.shape[1]))

# for sc in Senarios:
#     Pd_sc[sc, :, :] = pi[sc] * Pd
#     Qd_sc[sc, :, :] = pi[sc] * Qd
#     Pd_sc[sc, :, 0] = Pd[:, 0]  # fix t=0
#     Qd_sc[sc, :, 0] = Qd[:, 0]

# # === Build model ===
# model = create_Stochastic_ACOPF(data, nT=nT, NumSc=NumSc)

# # === Populate scenario-dependent demands ===
# for b in Bus:
#     b_idx = Bus.index(b)
#     for t in T:
#         for sc in Senarios:
#             model.Pd[sc, b, t].value = Pd_sc[sc, b_idx, t]
#             model.Qd[sc, b, t].value = Qd_sc[sc, b_idx, t]

# # === Solve ===
# solver = SolverFactory('ipopt')
# results = solver.solve(model, tee=False)

# # === Extract results ===
# Pgt = np.zeros((NumSc, len(G), len(T)))

# for g_idx, g in enumerate(G):
#     for t in T:
#         for sc in Senarios:
#             Pgt[sc, g_idx, t] = model.Pg[g, t, sc].value

# # === Plot results for one generator (g_index = 0) ===
# print('Stochastic SC-OPF schedule (Pg[0])')

# plt.figure(figsize=(6, 3.5))
# g_index = 0

# for sc in Senarios:
#     plt.plot(T, Pgt[sc, g_index, :], label=f'Scenario {sc}', alpha=0.6)

# plt.plot(Pgt[:, g_index, :].mean(axis=0), linestyle='dashed',
#          linewidth=2, color='black', label='Mean')

# plt.title(f'Active Power Generation: Pg[{g_index}]')
# plt.xlabel('Time')
# plt.ylabel('Power (pu)')
# plt.legend(loc='best', fontsize='small')
# plt.grid(True)
# plt.tight_layout()
# plt.show()




# %% ADMM-based stochastic ACOPF

def create_XUpdate_Stochastic_ACOPF(data, nT=24, NumSc=10, rho=1, Ramp=True, Tlink=False):
    """
    Creates a Pyomo model for the x-update step in ADMM-based stochastic ACOPF.

    This model solves the first-stage subproblem (local updates per agent),
    minimizing the cost function with consensus enforcement (rho, lambda terms).
    This function is used for single case studies and execution time.


    Parameters:
        data (dict): Network and generation data.
        nT (int): Number of time steps.
        NumSc (int): Number of scenarios.
        rho (float): Penalty parameter in ADMM.
        Ramp (bool): Whether to include ramping constraints.
        Tlink (bool): If True, enforces ramping continuity with an external Pg0.

    Returns:
        model (ConcreteModel): Pyomo model for the x-update optimization.
    """

    # === Extract data
    Bus        = data['Bus']
    branch     = data['branch']
    Lines      = data['Lines']
    Gen_data   = data['Gen_data']
    G          = data['G']
    G2B        = data['G2B']
    vmin       = data['vmin']
    vmax       = data['vmax']

    Senarios = range(NumSc)
    T = range(nT)
    EndTime = nT - 1

    # === Define model
    model = ConcreteModel(name='XUpdate_ACOPF')

    # Decision variables
    model.Pg     = Var(G, T, bounds=(0, None))
    model.Qg     = Var(G, T, bounds=(None, None))
    model.V2     = Var(Bus, T, bounds=(0, None), initialize=1)
    model.L2     = Var(Lines, T, bounds=(0, None))
    model.Pflow  = Var(Lines, T, bounds=(None, None))
    model.Qflow  = Var(Lines, T, bounds=(None, None))
    model.OF     = Var(bounds=(0, None))

    # ADMM parameters
    model.lambdaP = Param(Senarios, G, initialize=0, mutable=True)
    model.PgZ     = Param(Senarios, G, initialize=0, mutable=True)

    # Load parameters (mutable)
    model.Pd = Param(Bus, T, initialize=0, mutable=True)
    model.Qd = Param(Bus, T, initialize=0, mutable=True)
    if Tlink:
        model.Pg0 = Param(G, mutable=True)

    # === Constraints

    # Power balance
    def eqPbalance(model, b, t):
        return (
            sum(model.Pg[g, t] for g, bb in G2B.select('*', b)) - model.Pd[b, t]
            == sum(model.Pflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*'))
             - sum(model.Pflow[l, i, j, t] - branch.loc[(l, i, j)]['r'] * model.L2[l, i, j, t]
                   for l, i, j in Lines.select('*', '*', b))
        )
    model.eqPbalance = Constraint(Bus, T, rule=eqPbalance)

    def eqQbalance(model, b, t):
        return (
            sum(model.Qg[g, t] for g, bb in G2B.select('*', b)) - model.Qd[b, t]
            == sum(model.Qflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*'))
             - sum(model.Qflow[l, i, j, t] - branch.loc[(l, i, j)]['x'] * model.L2[l, i, j, t]
                   for l, i, j in Lines.select('*', '*', b))
        )
    model.eqQbalance = Constraint(Bus, T, rule=eqQbalance)

    # SOCP relaxation
    model.eqSij = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2 <= m.V2[i, t] * m.L2[l, i, j, t])

    model.eqSij_V = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        m.V2[i, t] - m.V2[j, t] ==
        -branch.loc[(l, i, j)]['z2'] * m.L2[l, i, j, t] +
         2 * (branch.loc[(l, i, j)]['r'] * m.Pflow[l, i, j, t]
            + branch.loc[(l, i, j)]['x'] * m.Qflow[l, i, j, t]))

    # Voltage limits
    model.eqVmax = Constraint(Bus, T, rule=lambda m, i, t: m.V2[i, t] <= vmax**2)
    model.eqVmin = Constraint(Bus, T, rule=lambda m, i, t: m.V2[i, t] >= vmin**2)

    # Line limits
    model.eqPijmax = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2 <= branch.loc[(l, i, j)]['limit']**2)

    model.eqPijmin = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        -branch.loc[(l, i, j)]['limit']**2 <= m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2)

    # Generator limits
    model.eqPgmax = Constraint(G, T, rule=lambda m, g, t: m.Pg[g, t] <= Gen_data.loc[g]['Pmax'])
    model.eqPgmin = Constraint(G, T, rule=lambda m, g, t: m.Pg[g, t] >= Gen_data.loc[g]['Pmin'])
    model.eqQgmax = Constraint(G, T, rule=lambda m, g, t: m.Qg[g, t] <= Gen_data.loc[g]['Qmax'])
    model.eqQgmin = Constraint(G, T, rule=lambda m, g, t: m.Qg[g, t] >= Gen_data.loc[g]['Qmin'])

    # Ramping constraints
    if Ramp:
        def eqRU(model, g, t):
            if t != EndTime:
                return model.Pg[g, t + 1] - model.Pg[g, t] <= Gen_data.loc[g]['RampUp']
            return Constraint.Skip
        model.eqRU = Constraint(G, T, rule=eqRU)

        def eqRD(model, g, t):
            if t != EndTime:
                return model.Pg[g, t] - model.Pg[g, t + 1] <= Gen_data.loc[g]['RampDn']
            return Constraint.Skip
        model.eqRD = Constraint(G, T, rule=eqRD)

        if Tlink:
            model.eqPg0up = Constraint(G, rule=lambda m, g: m.Pg[g, T[0]] - m.Pg0[g] <= Gen_data.loc[g]['RampUp'])
            model.eqPg0dn = Constraint(G, rule=lambda m, g: m.Pg0[g] - m.Pg[g, T[0]] <= Gen_data.loc[g]['RampDn'])

    # === Objective: cost + ADMM quadratic term
    model.eqOF = Constraint(expr=model.OF >= sum(
        Gen_data.loc[g]['b'] * model.Pg[g, t] +
        0.0001 * Gen_data.loc[g]['b'] * model.Qg[g, t]**2
        for g in G for t in T ) + sum(
        model.lambdaP[sc, g] * (model.Pg[g, 0] - model.PgZ[sc, g]) +
        0.5 * rho * (model.Pg[g, 0] - model.PgZ[sc, g])**2
        for g in G for sc in Senarios))

    model.obj = Objective(expr=model.OF, sense=pyo.minimize)

    return model


# %%

def Solve_Xupdate_ACOPF(data, modelX, Pd_sc_array, PgZ, lambda_s):
    """
    Solves the X-update subproblem in ADMM-based stochastic ACOPF using scenario-averaged demand.

    Parameters:
        data (dict): Dataset with Bus, G, and demand arrays.
        modelX (ConcreteModel): Pyomo model created by create_XUpdate_Stochastic_ACOPF().
        Pd_sc_array (np.ndarray): Demand scenarios (NumScenarios x NumBuses x Time).
        PgZ (np.ndarray): Consensus variable from ADMM (NumScenarios x NumGenerators).
        lambda_s (np.ndarray): Dual variables (NumScenarios x NumGenerators).

    Returns:
        dict: Contains 'Pg' (generator output array) and 'time' (solver runtime in seconds).
    """

    Bus      = data['Bus']
    G        = data['G']
    Pd_array = data['Pd_array']
    Qd_array = data['Qd_array']

    nT     = Pd_sc_array.shape[2]
    NumSc  = Pd_sc_array.shape[0]

    solver = SolverFactory('gurobi')

    # === Estimate Qd scenarios using fixed Q/P ratio
    Qd_sc_array = Pd_sc_array.copy()
    for b_ind in range(len(Bus)):
        if Pd_array[b_ind, 0] != 0:
            Q_ratio = Qd_array[b_ind, 0] / Pd_array[b_ind, 0]
            Qd_sc_array[:, b_ind, :] = Pd_sc_array[:, b_ind, :] * Q_ratio

    # === Set average demand in model
    for b_ind, b in enumerate(Bus):
        for t in range(nT):
            modelX.Pd[b, t].value = Pd_sc_array[:, b_ind, t].mean()
            modelX.Qd[b, t].value = Qd_sc_array[:, b_ind, t].mean()

    # === Set ADMM parameters
    for sc in range(NumSc):
        for g_ind, g in enumerate(G):
            modelX.PgZ[sc, g].value     = PgZ[sc, g_ind]
            modelX.lambdaP[sc, g].value = lambda_s[sc, g_ind]

    # === Solve model
    results = solver.solve(modelX, tee=False)
    ex_time = results['Solver'][0]['Time']

    # === Extract generation result
    Pgt_array = np.zeros((len(G), nT))
    for g_ind, g in enumerate(G):
        for t in range(nT):
            Pgt_array[g_ind, t] = modelX.Pg[g, t].value

    return {
        'Pg': Pgt_array,
        'time': ex_time
    }

    

# %%
def create_ZUpdate_Stochastic_ACOPF(data, nT=24, NumSc=10, rho=1,
                                    Ramp=True, Tlink=False):
    """
    Creates the Pyomo model for the Z-update step in ADMM-based stochastic ACOPF.

    This model updates the global consensus variable (Z) by minimizing the ADMM-augmented
    objective subject to power flow and network constraints, using average PgX from scenarios.

    This function is used for single case studies and execution time.


    Parameters:
        data (dict): Dataset including buses, generators, and power system topology.
        nT (int): Number of time steps.
        NumSc (int): Number of scenarios (needed for loop logic).
        rho (float): Penalty parameter in ADMM.
        Ramp (bool): Whether to enforce generator ramping constraints.
        Tlink (bool): Whether to include coupling to a fixed initial Pg0.

    Returns:
        model (ConcreteModel): Pyomo model for the Z-update optimization.
    """

    # === Extract data
    Bus        = data['Bus']
    branch     = data['branch']
    Lines      = data['Lines']
    Gen_data   = data['Gen_data']
    G          = data['G']
    G2B        = data['G2B']
    vmin       = data['vmin']
    vmax       = data['vmax']

    T = range(nT)
    EndTime = nT - 1

    # === Define model
    model = ConcreteModel(name='ZUpdate_ACOPF')

    # Decision variables
    model.Pg    = Var(G, T, bounds=(0, None))
    model.Qg    = Var(G, T, bounds=(None, None))
    model.V2    = Var(Bus, T, bounds=(0, None), initialize=1)
    model.L2    = Var(Lines, T, bounds=(0, None))
    model.Pflow = Var(Lines, T, bounds=(None, None))
    model.Qflow = Var(Lines, T, bounds=(None, None))
    model.OF    = Var(bounds=(0, None))

    # ADMM-related parameters
    model.lambdaP = Param(G, initialize=0, mutable=True)
    model.PgX     = Param(G, initialize=0, mutable=True)

    # Load profiles (fixed)
    model.Pd = Param(Bus, T, initialize=0, mutable=True)
    model.Qd = Param(Bus, T, initialize=0, mutable=True)

    if Tlink:
        model.Pg0 = Param(G, mutable=True)

    # === Constraints

    # Power balance
    def eqPbalance(model, b, t):
        return (
            sum(model.Pg[g, t] for g, bb in G2B.select('*', b)) - model.Pd[b, t]
            == sum(model.Pflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*'))
             - sum(model.Pflow[l, i, j, t] - branch.loc[(l, i, j)]['r'] * model.L2[l, i, j, t]
                   for l, i, j in Lines.select('*', '*', b))
        )
    model.eqPbalance = Constraint(Bus, T, rule=eqPbalance)

    def eqQbalance(model, b, t):
        return (
            sum(model.Qg[g, t] for g, bb in G2B.select('*', b)) - model.Qd[b, t]
            == sum(model.Qflow[l, i, j, t] for l, i, j in Lines.select('*', b, '*'))
             - sum(model.Qflow[l, i, j, t] - branch.loc[(l, i, j)]['x'] * model.L2[l, i, j, t]
                   for l, i, j in Lines.select('*', '*', b))
        )
    model.eqQbalance = Constraint(Bus, T, rule=eqQbalance)

    # SOCP power flow
    model.eqSij = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2 <= m.V2[i, t] * m.L2[l, i, j, t])

    model.eqSij_V = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        m.V2[i, t] - m.V2[j, t] ==
        -branch.loc[(l, i, j)]['z2'] * m.L2[l, i, j, t] +
         2 * (branch.loc[(l, i, j)]['r'] * m.Pflow[l, i, j, t]
            + branch.loc[(l, i, j)]['x'] * m.Qflow[l, i, j, t]))

    # Voltage bounds
    model.eqVmax = Constraint(Bus, T, rule=lambda m, i, t: m.V2[i, t] <= vmax ** 2)
    model.eqVmin = Constraint(Bus, T, rule=lambda m, i, t: m.V2[i, t] >= vmin ** 2)

    # Line flow bounds
    model.eqPijmax = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2 <= branch.loc[(l, i, j)]['limit']**2)

    model.eqPijmin = Constraint(Lines, T, rule=lambda m, l, i, j, t:
        -branch.loc[(l, i, j)]['limit']**2 <= m.Pflow[l, i, j, t]**2 + m.Qflow[l, i, j, t]**2)

    # Generator capacity limits
    model.eqPgmax = Constraint(G, T, rule=lambda m, g, t: m.Pg[g, t] <= Gen_data.loc[g]['Pmax'])
    model.eqPgmin = Constraint(G, T, rule=lambda m, g, t: m.Pg[g, t] >= Gen_data.loc[g]['Pmin'])
    model.eqQgmax = Constraint(G, T, rule=lambda m, g, t: m.Qg[g, t] <= Gen_data.loc[g]['Qmax'])
    model.eqQgmin = Constraint(G, T, rule=lambda m, g, t: m.Qg[g, t] >= Gen_data.loc[g]['Qmin'])

    # Ramping constraints
    if Ramp:
        def eqRU(model, g, t):
            if t != EndTime:
                return model.Pg[g, t + 1] - model.Pg[g, t] <= Gen_data.loc[g]['RampUp']
            return Constraint.Skip
        model.eqRU = Constraint(G, T, rule=eqRU)

        def eqRD(model, g, t):
            if t != EndTime:
                return model.Pg[g, t] - model.Pg[g, t + 1] <= Gen_data.loc[g]['RampDn']
            return Constraint.Skip
        model.eqRD = Constraint(G, T, rule=eqRD)

        if Tlink:
            model.eqPg0up = Constraint(G, rule=lambda m, g: m.Pg[g, T[0]] - m.Pg0[g] <= Gen_data.loc[g]['RampUp'])
            model.eqPg0dn = Constraint(G, rule=lambda m, g: m.Pg0[g] - m.Pg[g, T[0]] <= Gen_data.loc[g]['RampDn'])

    # === Objective: local cost + ADMM penalty
    model.eqOF = Constraint(expr=model.OF >= sum(
        Gen_data.loc[g]['b'] * model.Pg[g, t] +
        0.0001 * Gen_data.loc[g]['b'] * model.Qg[g, t]**2
        for g in G for t in T
    ) + sum(
        model.lambdaP[g] * (model.PgX[g] - model.Pg[g, 0]) +
        0.5 * rho * (model.PgX[g] - model.Pg[g, 0])**2
        for g in G
    ))

    model.obj = Objective(expr=model.OF, sense=pyo.minimize)

    return model



# %% Main ADMM function for Stochastic ACOPF

def Solve_ADMM_Stochastic_ACOPF(
    data,
    k_iter=10,
    modelX=None,
    modelZ=None,
    lambda_initial=None,
    PgX_initial=None,
    PgZ_initial=None,
    rho=0.1,
    lambda_std=0.3,
    exploration=False,
    explore_iter=0,
    exp_decay=0.1,
    Ramp=True,
    DemandInstnace=None,
    print_result=False
):
    """
    Solves the stochastic ACOPF problem using ADMM decomposition.

    Parameters:
        data (dict): Input data (buses, generators, lines, loads, etc.)
        k_iter (int): Number of ADMM iterations.
        modelX (ConcreteModel): Pyomo model for the X-update.
        modelZ (ConcreteModel): Pyomo model for the Z-update.
        lambda_initial (np.ndarray): Initial dual variables (NumScenarios x NumGenerators).
        PgX_initial (np.ndarray): Initial shared generator decision (G,).
        PgZ_initial (np.ndarray): Initial scenario generator decisions (NumSc x G x T).
        rho (float): ADMM penalty parameter.
        lambda_std (float): Std of initial lambda if random.
        exploration (bool): Enable stochastic exploration in lambda update.
        explore_iter (int): Number of iterations to explore before convergence.
        exp_decay (float): Decay rate for exploration probability.
        Ramp (bool): Use ramp constraints.
        DemandInstnace (np.ndarray): Demand matrix (Scenarios x Buses x T).
        print_result (bool): Show convergence plots and logs.

    Returns:
        dict: Includes PgZ, PgX, residuals, dual variables, and time metrics.
    """

    # Extract data
    Bus = data['Bus']
    G = data['G']
    Pd_array = data['Pd_array']
    Qd_array = data['Qd_array']

    # === Setup demand
    if DemandInstnace is None:
        raise ValueError("No demand input provided!")

    Pd_sc_array = DemandInstnace.copy()
    Qd_sc_array = Pd_sc_array.copy()

    for b_ind in range(len(Bus)):
        if Pd_array[b_ind, 0] != 0:
            Qd_sc_array[:, b_ind, :] = Pd_sc_array[:, b_ind, :] * (Qd_array[b_ind, 0] / Pd_array[b_ind, 0])

    nT = Pd_sc_array.shape[2]
    NumSc = Pd_sc_array.shape[0]
    T = range(nT)
    Senarios = range(NumSc)
    k_iter_list = range(k_iter)

    # === Initialize
    PgX = np.zeros(len(G))
    PgZ = np.zeros((NumSc, len(G), len(T)))
    PgX_k = np.zeros((k_iter, len(G), len(T)))
    PgZ_k = np.zeros((k_iter, NumSc, len(G), len(T)))
    lambda_s = lambda_initial if lambda_initial is not None else np.random.normal(0, lambda_std, (NumSc, len(G)))
    lambda_k = np.zeros((k_iter, NumSc, len(G)))
    Time_X_k = np.zeros(k_iter)
    Time_Z_k = np.zeros((k_iter, NumSc))
    residual = {'rk': []}

    exp_mask = np.random.uniform(0, 1, size=(k_iter, NumSc, len(G)))
    exp_threshold = 1.0

    # === Models
    if modelX is None:
        modelX = create_XUpdate_Stochastic_ACOPF(data, NumSc=NumSc, nT=nT, rho=rho, Ramp=Ramp, Tlink=False)
    if modelZ is None:
        modelZ = create_ZUpdate_Stochastic_ACOPF(data, nT=nT, NumSc=NumSc, rho=rho, Ramp=Ramp, Tlink=False)

    if PgX_initial is not None:
        PgX = PgX_initial.copy()
    if PgZ_initial is not None:
        PgZ = PgZ_initial.copy()

    # === Initialize solver
    solver = SolverFactory('gurobi')

    # === Average demands for X-model
    for b_ind, b in enumerate(Bus):
        for t in T:
            modelX.Pd[b, t].value = Pd_sc_array[:, b_ind, t].mean()
            modelX.Qd[b, t].value = Qd_sc_array[:, b_ind, t].mean()

    # === ADMM Iterations ===
    start_time = time.time()

    for k in k_iter_list:

        # === X-Update ===
        for sc in Senarios:
            for g_ind, g in enumerate(G):
                modelX.PgZ[sc, g].value = PgZ[sc, g_ind, 0]
                modelX.lambdaP[sc, g].value = lambda_s[sc, g_ind]

        results = solver.solve(modelX, tee=False)
        Time_X_k[k] = results['Solver'][0]['Time']

        for g_ind, g in enumerate(G):
            PgX[g_ind] = modelX.Pg[g, 0].value
            for t in T:
                PgX_k[k, g_ind, t] = modelX.Pg[g, t].value

        # === Z-Update ===
        for sc in Senarios:
            if print_result:
                print(f'ADMM iter {k} - scenario {sc}', end='\r')

            for b_ind, b in enumerate(Bus):
                for t in T:
                    modelZ.Pd[b, t].value = Pd_sc_array[sc, b_ind, t]
                    modelZ.Qd[b, t].value = Qd_sc_array[sc, b_ind, t]

            for g_ind, g in enumerate(G):
                modelZ.PgX[g].value = PgX[g_ind]
                modelZ.lambdaP[g].value = lambda_s[sc, g_ind]

            results = solver.solve(modelZ, tee=False)
            Time_Z_k[k, sc] = results['Solver'][0]['Time']

            for g_ind, g in enumerate(G):
                PgZ[sc, g_ind, :] = [modelZ.Pg[g, t].value for t in T]
                lambda_s[sc, g_ind] += rho * (PgX[g_ind] - modelZ.Pg[g, 0].value)

        # === Lambda Exploration (Optional) ===
        if exploration and k < explore_iter:
            exp_threshold -= exp_decay
            lambda_s_rand = np.random.normal(0, lambda_std, (NumSc, len(G)))
            for sc in Senarios:
                for g in range(len(G)):
                    if exp_mask[k, sc, g] <= exp_threshold:
                        lambda_s[sc, g] = lambda_s_rand[sc, g]  # Explore

        # === Residual Update ===
        PgZ_k[k] = PgZ
        lambda_k[k] = lambda_s
        rk_now = np.square(np.abs(PgX - PgZ[:, :, 0])).mean()
        residual['rk'].append(rk_now)

    end_time = time.time()
    ex_time = end_time - start_time

    # === Print & Plot
    if print_result:
        print(f"\nADMM finished in {ex_time:.2f} sec")

        plt.figure(figsize=(5, 3))
        plt.plot(range(k_iter), residual['rk'])
        plt.title('Residuals')
        plt.xlabel('Iteration k')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(5, 3))
        for sc in Senarios:
            plt.plot(PgZ[sc, 0, :], label=f"Sc {sc}", alpha=0.6)
        plt.plot(PgZ[:, 0, :].mean(axis=0), linestyle='dashed', linewidth=2, color='black', label='mean')
        plt.title('Generator 0 Profile across Scenarios')
        plt.legend()
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
"""
Toy Example: Solving a Stochastic ACOPF Problem using ADMM

This example demonstrates how to:
1. Load IEEE 14-bus system data
2. Generate naive stochastic demand scenarios
3. Run the ADMM-based stochastic ACOPF solver
"""

# === Step 1: Load data for the IEEE 14-bus system ===
sys = 14
nT = 6  # Time horizon (6 periods)
data = read_data_ACOPF(File=f'IEEE_{sys}_bus_Data_PGLib_ACOPF.xlsx', DemFactor=1.0, print_data=False)

Bus = data['Bus']
G = data['G']
Pd_array = data['Pd_array']

# === Step 2: Create stochastic demand scenarios (NumSc scenarios) ===
NumSc = 5
Senarios = range(NumSc)
T = range(nT)

Pd_sc_array = np.zeros((NumSc, Pd_array.shape[0], Pd_array.shape[1]))
np.random.seed(2023)
pi = np.random.uniform(0.9, 1.1, size=NumSc)

for sc in Senarios:
    Pd_sc_array[sc, :, :] = pi[sc] * Pd_array
    Pd_sc_array[sc, :, 0] = Pd_array[:, 0]  # First period fixed 

# === Step 3: Solve stochastic ACOPF using ADMM ===
result = Solve_ADMM_Stochastic_ACOPF(
    data,
    k_iter=10,
    DemandInstnace=Pd_sc_array[:, :, :nT],
    print_result=True,
    exploration=True,     # Enable stochastic exploration of duals
    explore_iter=5,       # Explore for the first 5 iterations
    exp_decay=0           # No decay in exploration threshold
)


# End


