# -*- coding: utf-8 -*-
"""
Last update: Sep. 2, 2020

@author: Asieh Abolpour Mofrad

This code is used for simulation results reported in an article entitled:
    ''Enhanced Equivalence Projective Simulation:
        a Framework for Modeling Formation of Stimulus Equivalence Classes"
        in Neural Computation, MIT Press.

"""

import initialization
import initialization_detail
import environment as env
import agent as agn
import interaction as intrc
import pdb

environment_detail = initialization_detail.environment_details()
environment_parameter, agent_parameter = initialization.config()

# Give the file_name to just plot a previousely saved simulation
file_name = None

if file_name == None:
    agent = agn.Agent(agent_parameter)
    environment = env.Environment(environment_parameter)
    interaction = intrc.Interaction(agent, environment, agent_parameter,
                                                        environment_parameter)
    interaction.run_save()
    file_name = interaction.file_name
    print(file_name)

plt_ = intrc.Plot_results(file_name)
plt_.showResults()
plt_.print_setting()