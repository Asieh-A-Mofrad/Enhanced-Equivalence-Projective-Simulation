# -*- coding: utf-8 -*-
"""
Last update: Sep. 2, 2020

@author: Asieh Abolpour Mofrad

Initialization values

This code is used for simulation results reported in an article entitled:
    ''Enhanced Equivalence Projective Simulation:
        a Framework for Modeling Formation of Stimulus Equivalence Classes"
        in Neural Computation, MIT Press.

"""

def config():
    return environment_parameters(), agent_parameters()

def environment_parameters():

    """
    The initial setting data. which are positive digits.
    """

    environment_parameter = {
                             "experiment_ID": [00],
                             "environment_ID": [4],
                             "max_trial": [10000],
                             "num_agents": [1],
                             "size_action_set": [3]
                             }

    return environment_parameter

def agent_parameters():

    """
    This is an initialization for agent parameters based on the agent type.

        - network_enhancement: True or False; if True: we use the localized network
        (adoubly stochastic matrix) and Symmetric Network Enhancement (SNE); 
        if False: we use the probability matrix and Directed Network Enhancement (DNE)

        - gamma_damping: float between 0 and 1, controls forgetting/damping
        of h-values

        - beta_h: float >=0, probabilities are proportional to
        exp(beta_h*h_value). is used for converting h-values to probabilities
        during training and to generate the Network enhancement input

        - beta_t:  float >=0, controls agent’s function in a trial at test phase (derived relation)

        - 0 < K <= 1 is the parameter for symmetry relation. K=1 means that
        the relations are bidirectional and network is symmetric at the end of 
        the training phase.

        alpha: the regularization parameter

    """

    agent_parameter = {
                       "network_enhancement": [False],
                       "beta_h": [0.07],
                       "beta_t": [8],
                       "K": [1],
                       "gamma_damping": [0.001],
                       "alpha": [0.8]
                       }

    return agent_parameter

