# Enhanced Equivalence Projective Simulation (E-EPS)

This code is supposed to simulate the "enhanced equivalence projective simulation" model that uses 
[Projective Simulation (PS)](https://projectivesimulation.org) framework for computationaly 
modeling equivalence class formation in Behavior analysis. To model the formation of *derived relations*, the model uses [Network Enhancement (NE)](https://www.nature.com/articles/s41467-018-05469-x).
This code is used for producing results in paper entitled:

**Mofrad, A. A., Yazidi, A., Mofrad, S. A., Hammer, H. L., & Arntzen, E. (2020) Enhanced Equivalence Projective Simulation: a Framework for Modeling Formation of Stimulus Equivalence Classes** (Accepted in Neural Computation)

which is an extention of the "equivalence projective simulation" model which is introduced in [Equivalence Projective Simulation (EPS)]{https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01274}.

Please cite the paper if you use the E-EPS model or the code in this repository. 

E-EPS computationally models formation of equivalence classes in Behavior Analysis. Briefly, in an arbitrary MTS experiment, first
some arbitrary relations say A1-B1, A2-B2, B1-C1, B2-C2, are trained through a matching-to-sample procedure. Then, after agent masters these relations, i.e. answering correctly to say 90% of the trials in a block of certain size of trials, the agent will be tested for **derived** relations, say A1-C1, C2-A2. 

## Getting Started
### Prerequisites

This code has been tested with Python 3.6.5 (Anaconda)

The program might crash if the initial values are not valid

## Process summary

Before running the "main.py", you can change the initial values in initialization.py\
 By initialization and running main.py:

- An agent, an environment, an interaction, and possibly an interface object will be initializes

- The results to be plotted will be saved in a pickle file in the "results" folder.

- The figures will be shown in new windows.

- Results of a previously simulated data can be accessed by its ID.  

- Percept (sample stimulus) and action_set (comparison stimuli) belong to two different categories. 
 
- When blocks are constructed by environment, the order of comparison stimuli and the order of trials in the block are random.

To see the results for a previous simulation, change file_name= None to a file name in the results folder, say
```python 
filename= 'results/Env_4_ExpID_3.p'
```
- Other parameters of environment and agent, can be changed in **initialization.py**:

## What initial values means

### For environment:
#### (In initialization.py)

"environment_parameter" can be updated:
 ```python 
 environment_parameter = {
                           "experiment_ID": [0],
                           "environment_ID": [1],
                           "max_trial": [10000],
                           "num_agents": [1000],
                           "size_action_set": [3]
                           }
```
where: 

**experiment_ID**:\
A positive number which is used for saving the results in the results folder. The file name would be: 
'Env_**environment_ID**_ExpID_**experiment_ID**'
    
**environment_ID**:\
Is the key to the defult ID numbers, that adress the details of a protocol

**max_trial**:\
The maximum number of allowed trials in the training phase. If the learning is so slow due to the parameters, the training might not be finished

**num_agents**:\
The number of participants (agents); The program will be repeated for num_agents times and the final result would be the average

**size_action_set**:\
The number of comparison stimuli in the experiment. This must be at least 2 and at most the num_classes


### Adding new Environment Settings:
#### (In initialization_detail.py)
New experiments can be added by defining new environment_IDs to the defaults: 
```python
environment_parameters_detail = {
    1: { 
        "num_classes":3,
        "training_order": {...}
        "plot_blocks": {...}
        "plot_blocks_ID": {...}
        "mastery_training":  0.9 },
    2: {...
    ...
    10: {
```


**num_classes**:\
Is the number of classes in the experiment. It must be compatible with training_order

**training_order**:\
The structure of training and blocks in dict() format. There are two ways to initialize it though:

1. let
```python
num_classes=4,
training_order={ \
                1: [('A', 'B', 30)], \
                2: [('B', 'C', 30)], \
                3: [('D', 'C', 30)] \
                }
``` 
and in initialization.py, we set 
```python
"size_action_set": [3]
```
The key values 1, 2, 3 shows the order of training. As an example, the first block contains 30 trials of relations with A1, A2, A3, or A4 as the sample stimuli and three comparison stimuli from B1, B2, B3, and B4. Note that the correct choice must be among the comparison stimuli, say for A2, (B1, B3, B4) is not a valid action set/comparison stimuli. Moreover, that number of trials at each block must be a multiple of num_classes. Since, the model produce the same number of trial for each particular pair. 
 
2. let num_classes=2, and size_action_set=2, and
```python
     training_order={ \
                     1:[('A1', 'B1', 10)],\
                     2:[('A2', 'B2', 10)],\
                     3:[('A1', 'B1', 5),('A2', 'B2', 5)],\
                     4:[('A1', 'C1', 10)],\
                     5:[('A2', 'C2', 10)],\
                     6:[('A1', 'C1', 5),('A2', 'C2', 5)],\
                     7:[('A1', 'B1', 2),('A2', 'B2', 2),('A1', 'C1', 2),('A2', 'C2', 2)]\
                    }
 ```           
 
In this case, the desired relation to be trained in each block and the number of its repetition is determined. The num_classes must be compatible with the provided relations. The above training_order means after mastery of relation A1-B1 in blocks of 10 trials, A2-B2 will be trained, then a block of mixed A1-B1 and A2-B2. Next, A1-C1, then A2-C2 and then a mixed block of A1-C1, A2-C2. Finally, all the trained relations construct a block and by passing the mastery criterion, the training phase will be finished.                
    In order to preperation of the results, two other dict() needs to be set in accordance with the training_order:
    
  - **plot_blocks**: \
That is an option for desired combination of relations, say:
```python        
   plot_blocks= {\
                'Direct':['AB', 'BC', 'DC'],\
                'Derived':['BA', 'CB', 'CD', 'AC', 'CA', 'BD', 'DB', 'AD', 'DA']\
                 }
  ```            
where 'AB' means all possible relations between the two categories say A1-B1, A2-B2, A3-B3, etc. One must set it as empty dict() to remove this extra plot. 
```python        
        "plot_blocks_ID": {'relatin_type':['Direct','Derived']}
```
**mastery_training**:\
A value between 0 and one that shows the mastery criteria. $0.9$ means $90\%$ correct choices in a block.  

### For agent:
#### (In initialization.py)

```python
agent_parameter = {
                   "network_enhancement": [False],
                   "beta_h": [0.07],
                   "beta_t": [8],
                    "K": [1],
                    "gamma_damping": [0.001],
                    "alpha": [0.8]
                    }
```

**network_enhancement**:\
Could be ['True', 'False']. If 'True', retrieval is based on *Symmetric Network Enhancement (SNE)*. If 'False', retrieval is based on *Directed Network Enhancement (DNE)*

**gamma_damping**:\
A float number between 0 and 1 which controls forgetting/damping of h-values. The closser to zero, the less forgetting and the closer to one, the less learning/memory.

**beta_h**:\
float >=0, probabilities are proportional to exp(beta_softmax*h_value) and finding the appropriate value is very important. Being used for converting h-values to probabilities during training and to generate the Network enhancement input. In general, its higher value, increases the chance of a connection with the largest h-value to be chosen.

**beta_t**:\
float >=0, controls agent's function in a trial at test phase (derived relation).

**K**:\
Is a positive value for symmetry relation. K=1 means that
 the relations are bidirectional and network is symmetric at the end of the training phase.

**alpha**:\
The regularization parameter


## License

MIT license (MIT-LICENSE or http://opensource.org/licenses/MIT)
