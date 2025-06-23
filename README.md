# Learning-Accelerated ADMM for Stochastic Power System Scheduling with Numerous Scenarios

**Author:** Ali Rajaei  
**Affiliation:** Delft-AI Energy Lab, Department of Electrical Sustainable Energy, Delft University of Technology, the Netherlands  
**Contact:** a.rajaei@tudelft.nl  
**Date:** April 2025  

This repository accompanies the research paper:

> Rajaei, Ali, Olayiwola Arowolo, and Jochen L. Cremer.  
> ["Learning-Accelerated ADMM for Stochastic Power System Scheduling with Numerous Scenarios."](https://ieeexplore.ieee.org/abstract/document/10971244/)  
> *IEEE Transactions on Sustainable Energy*, 2025.

---

## 📄 Abstract

The increasing share of uncertain renewable energy sources (RES) in power systems necessitates new efficient approaches for the two-stage stochastic multi-period AC optimal power flow (St-MP-OPF) optimization. The computational complexity of St-MP-OPF, particularly with AC constraints, grows exponentially with the number of uncertainty scenarios and the time horizon. This complexity poses significant challenges for large-scale transmission systems that require numerous scenarios to capture RES stochasticities.

This paper introduces a scenario-based decomposition of the St-MP-OPF based on the alternating direction method of multipliers (ADMM). Additionally, it proposes a machine learning-accelerated ADMM approach (ADMM-ML), facilitating rapid and parallel computations of numerous scenarios with extended time horizons. Within this approach, a recurrent neural network approximates the ADMM sub-problem optimization and predicts wait-and-see decisions for uncertainty scenarios, while a master optimization determines here-and-now decisions. A hybrid approach is also developed, which uses ML predictions to warm-start the ADMM algorithm, combining the computational efficiency of ML with the feasibility and optimality guarantees of optimization methods.

The numerical results on the 118-bus and 1354-bus systems show that the proposed ADMM-ML approach solves the St-MP-OPF with 3–4 orders of magnitude speed-ups, while the hybrid approach provides a balance between speed-ups and optimality.

---

## 📌 Repository Overview

This repository contains:
- ✅ Pyomo-Gurobi implementation of the Stochastic AC multi-period OPF  
- ✅ Scenario-based decomposition using ADMM  
- ✅ ML model for approximating ADMM subproblems  
- ✅ Training and evaluation pipelines for ADMM-ML and hybrid solutions  
- ✅ Training data generation with $\epsilon$-greedy exploration
  

---

## 📎 Paper Link

[🔗 IEEE Xplore - View Paper](https://ieeexplore.ieee.org/abstract/document/10971244/)
[🔗 PowerTech 2025 Presentation](https://github.com/TU-Delft-AI-Energy-Lab/ADMM_ML_StMPOPF/blob/main/PowerTech_Presentation.pdf)

