# Enhanced Equivalence Projective Simulation

This code is supposed to simulate the "enhanced equivalence projective simulation" model that uses 
[Projective Simulation (PS)](https://projectivesimulation.org) framework for computationaly 
modeling equivalence class formation in Behavior analysis. To model the formation of *derived relations*, the model uses [Network Enhancement (NE)](https://www.nature.com/articles/s41467-018-05469-x).
This code is used for producing results in paper entitled:

**Enhanced Equivalence Projective Simulation: a Framework for Modeling Formation of Stimulus Equivalence Classes** (Accepted in Neural Computation)

and is an extention of the "equivalence projective simulation" model which is introduced in

**Mofrad, A. A., Yazidi, A., Hammer, H. L., & Arntzen, E. (2020). Equivalence Projective Simulation as a Framework for Modeling Formation of Stimulus Equivalence Classes. Neural Computation, 32(5), 912-968.**


- Overally, the simulation is to firat train some relations say A1-B1, A2-B2, B1-C1, B2-C2, through a matching-to-sample procedure. Then, after mastery of agent in these relations, i.e. answering correctly to say 90% of the trials in a block of certain size of trials, the agent will be tested.
- The test could be the relations that are trained directly, or some relations that are not explicitly traind, say **derived** relations.
- A protocol is a road map to which relations must be trained, what is the criteria for mastery, what is the testing relations, etc. 
- In the model, the details of protocol must be changed in initialization_detail.py, but more general parameters can be changed in initialization.py 
- Interaction between agent and Environment, is by passing a matching-to-sample trial to the agent, passing the agent choice to the Environment, and the feedback/reward -in the training phase- to the agent.  

## Getting Started
### Prerequisites

This code has been tested with Python 3.6.5 (Anaconda)

The program might crash if the initial values are not valid

## Files details

Before running the "main.py", you can change the initial values in initialization.py
The Initialization_detail.py is for adding new experiments.
The running process is as follows:
An agent, an environment, and an intreaction object will be initializes
The results to be ploted will be saved in a pickle file in the "results" folder.
The figures will be shown in new windows.
Results of a previousely simulated data can be accessed by its ID.

## What initial values means

For environment:
(In initialization.py)
experiment_ID: A positive number which is used for saving the results in the results folder. The file name would be: 'Env_environment_IDExpIDexperiment_ID'
environment_ID: Is the key to the defult ID numbers, that adress the details of a protocol
max_trial: The maximum number of allowed trials in the training phase. If the learning is so slow due to the parameters, the training might not be finished
num_agents: The number of participants (agents); The program will be repeated for num_agents times and the final result would be the average
size_action_set: The number of comparison stimuli in the experiment. This must be at least 2 and at most the num_classes
(In initialization_detail.py)
num_classes: Is the number of classes in the experiment. It must be compatible with training_order
training_order: The structure of training and blocks in dict() format. There are two ways to initialize it though:
First: let num_classes=4, and size_action_set=3, and
training_order={
               1: [('A', 'B', 40)], 
               2: [('B', 'C', 40)], 
               3: [('D', 'C', 40)] 
               }.
The numbers 1,2,3 shows the order of training. As an example, the first block contains 30 trials of relations with A1, A2, A3, or A4 as the sample stimuli and three comparison stimuli from B1, B2, B3, and B4. Note that the correct choice must be among the comparison stimuli, say for A2, (B1, B3, B4) is not a valid action set/comparison stimuli. Moreover that number of trials at each block must be a multiple of num_classes. Since, the model produce the same number of trial for each particular pair. 
Second: let num_classes=2, and size_action_set=2, and
training_order={
               1:[('A1', 'B1', 10)],
               2:[('A2', 'B2', 10)],
               3:[('A1', 'B1', 5),('A2', 'B2', 5)],
               4:[('A1', 'C1', 10)],
               5:[('A2', 'C2', 10)],
               6:[('A1', 'C1', 5),('A2', 'C2', 5)],
               7:[('A1', 'B1', 2),('A2', 'B2', 2),('A1', 'C1', 2),('A2', 'C2', 2)]
              }
This case, the desired relation to be trained in each block and the number of its repetition is determined. The num_classes must be compatible with the provided relations. The above training_order means after mastery of relation A1-B1 in blocks of 10 trials, A2-B2 will be trained, then a block of mixed A1-B1 and A2-B2. Next, A1-C1, then A2-C2 and then a mixed block of A1-C1, A2-C2. Finally, all the trained relations will make a block and by passing the mastery criteria, the training phase will be finished. 
In order to preperation of the results, two other dict() needs to be set in accordance with the training_order:
plot_blocks: That is an option for desired combination of relasions, say:
plot_blocks= {
              'Direct':['AB', 'BC', 'DC'],
              'Derived':['BA', 'CB', 'CD', 'AC', 'CA', 'BD', 'DB', 'AD', 'DA']}.
where 'AB' means all possible relations between the two categories say A1-B1, A2-B2, A3-B3, etc. One must set it as empty dict() to remove this extra plot. 
"plot_blocks_ID": {'relatin_type':['Direct','Derived']}
mastery_training: A value between 0 and one that shows the mastery criteria. 
0.9
0.9
means 
90%
90%
correct choices in a block. 
For agent:
(In initialization.py)
network_enhancement: Could be ['True', 'False']. If 'True', retrieval is based on Symmetric Network Enhancement (SNE). If 'False', retrieval is based on Directed Network Enhancement (DNE)
gamma_damping: A float number between 0 and 1 which controls forgetting/damping of h-values. The closser to zero, the less forgetting and the closer to one, the less learning/memory.
beta_h: float >=0, probabilities are proportional to exp(beta_softmax*h_value) and finding the appropriate value is very important. Being used for converting h-values to probabilities during training and to generate the Network enhancement input. In general, its higher value, increases the chance of a connection with the largest h-value to be chosen.
beta_t: float >=0, controls agent's function in a trial at test phase (derived relation).
K: Is a positive value for symmetry relation. K=1 means that the relations are bidirectional and network is symmetric at the end of the training phase.
alpha: The regularization parameter

## Assumptions/Constraints 

- Percept (sample stimulus) and action_set (comparison stimuli) belong to two different categories. 
-  When blocks are constructed by environment, the order of comparison stimuli and the order of trials in the block are random.

- There is two phases in the model, training with feedback and testing without feedback. In real experiments with human subjects during training, the feedback will be reduced to say $75\%$, $50\%$, and $25\%$ and it is to see if the participant remember the relations. In a synthetic and probabilistic model, the argument is not valid and we did not consider this case. 

## Author

* **Asieh Abolpour Mofrad** 
