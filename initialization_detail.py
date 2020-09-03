# -*- coding: utf-8 -*-
"""
Last update: Sep. 2, 2020

@author: Asieh Abolpour Mofrad

Initialization values details

This code is used for simulation results reported in an article entitled:
    ''Enhanced Equivalence Projective Simulation:
        a Framework for Modeling Formation of Stimulus Equivalence Classes"
        in Neural Computation, MIT Press.

"""

def environment_parameters_details(ID):

    details = environment_details()
    return details[ID]['num_classes'], details[ID]['training_order'], \
    details[ID]['plot_blocks'], details[ID]['plot_blocks_ID'], \
    details[ID]['mastery_training']

def environment_details():

    """
    Here the information for the environment will be provided.
    The number is what in the interaction file must be specified.
    plot_blocks shows representation for bar diagrams.
    """

    environment_parameters_detail = {
    1: { # This is the example at the paper
        "num_classes":3,
        "training_order": {
                           1:[('A','B',30)],
                           2:[('B','C',30)],
                           3:[('D','C',30)]
                           },

        "plot_blocks": {
                        'relation_type':{'Direct':['AB','BC','DC'],
                        'Derived':['BA','CB','CD','AC','CA','BD','DB','AD','DA']}
                        },

        "plot_blocks_ID": {
                           'relation_type':['Direct','Derived']
                           },

        "mastery_training":  0.9
        },

    10: { # This is the example at the paper
        "num_classes":5,
        "training_order": {
                           1:[('A','B',50)],
                           2:[('B','C',50)],
                           3:[('D','C',50)]
                           },

        "plot_blocks": {
                        'relation_type':{'Direct':['AB','BC','DC'],
                        'Derived':['BA','CB','CD','AC','CA','BD','DB','AD','DA']}
                        },

        "plot_blocks_ID": {
                           'relation_type':['Direct','Derived']
                           },

        "mastery_training":  0.95
        },


    2: { # This is the Sidman and Tailby experiment (1982)
        "num_classes": 3,
        "training_order": {
                           1:[('A1','B1',10),('A2','B2',10)],
                           2:[('A1','B1',10),('A3','B3',10)],
                           3:[('A2','B2',10),('A3','B3',10)],
                           4:[('A1','B1',10),('A2','B2',10),('A3','B3',10)],
                           5:[('A1','C1',10),('A2','C2',10)],
                           6:[('A1','C1',10),('A3','C3',10)],
                           7:[('A2','C2',10),('A3','C3',10)],
                           8:[('A1','C1',10),('A2','C2',10),('A3','C3',10)],
                           9: [('A1','B1',5),('A2','B2',5),('A3','B3',5),
                               ('A1','C1',5),('A2','C2',5), ('A3','C3',5)],
                           10: [('D1','C1',10),('D2','C2',10)],
                           11:[('D1','C1',10),('D3','C3',10)],
                           12: [('D2','C2',10),('D3','C3',10)],
                           13:[('D1','C1',10),('D2','C2',10),('D3','C3',10)],
                           14:[('A1','B1',5),('A2','B2',5),('A3','B3',5),
                               ('A1','C1',5),('A2','C2',5), ('A3','C3',5),
                               ('D1','C1',5),('D2','C2',5),('D3','C3',5)]
                           },

        "plot_blocks": {
                        'relation_type':{'Baseline':['AB','AC','DC'],
                        'Symmetry':['BA','CA','CD'],
                        'Equivalence':['BC','CB','BD','DB','AD','DA']}
                        },

        "plot_blocks_ID": {
                           'relation_type':['Baseline','Symmetry','Equivalence']
                          },

        "mastery_training": 0.9
        },


    3: { # This is the Devany et. al. experiment (1986)
        "num_classes": 2,
        "training_order": {
                           1:[('A1','B1',10)],
                           2:[('A2','B2',10)],
                           3:[('A1','B1',5),('A2','B2',5)],
                           4:[('A1','C1',10)],
                           5:[('A2','C2',10)],
                           6:[('A1','C1',5),('A2','C2',5)],
                           7: [('A1','B1',2),('A2','B2',2),('A1','C1',2),
                               ('A2','C2',2)]
                           },

        "plot_blocks": {
                        'relation_type':{'Baseline':['AB','AC'],'Symmetry':['BA','CA'],
                                        'Equivalence':['BC','CB']}
                        },

        "plot_blocks_ID": {'relation_type':['Baseline','Symmetry','Equivalence']},

        "mastery_training": 0.9
        },

# This is the Spencer and Chase experiment (1996)
    4: { #
        "num_classes": 3,
        "training_order": {
                           1:[('A','B',48)],
                           2:[('A','B',24),('B','C',24)],
                           3:[('A','B',12),('B','C',12),('C','D',24)],
                           4:[('A','B',9),('B','C',9),('C','D',9),('D','E',24)],
                           5:[('A','B',6),('B','C',6),('C','D',6),('D','E',6),
                              ('E','F',24)],
                           6:[('A','B',3),('B','C',3),('C','D',3),('D','E',6),
                              ('E','F',9),('F','G',24)],
                           7:[('A','B',3),('B','C',3),('C','D',3),('D','E',3),
                              ('E','F',3),('F','G',3)]
                           },

        "plot_blocks": {
                        'nodal_distance':{
                                          'Bsl':['AB','BC','CD','DE','EF','FG'],
                                          'Sym':['BA','CB','DC','ED','FE','GF'],
                                          '1-Tr':['AC','BD','CE','DF','EG'],
                                          '2-Tr':['AD','BE','CF','DG'],
                                          '3-Tr':['AE','BF','CG'],
                                          '4-Tr':['AF','BG'],
                                          '5-Tr':['AG'],
                                          '1-Eq':['CA','DB','EC','FD','GE'],
                                          '2-Eq':['DA','EB','FC','GD'],
                                          '3-Eq':['EA','FB','GC'],
                                          '4-Eq':['FA','GB'],
                                          '5-Eq':['GA']
                                          },
                        'relation_type':{
                                        'Baseline':['AB','BC','CD','DE','EF','FG'],
                                        'Symmetry':['BA','CB','DC','ED','FE','GF'],
                                        'Transivity':['AC','BD','CE','DF','EG',
                                                      'AD','BE','CF','DG','AE',
                                                      'BF','CG','AF','BG','AG'],
                                        'Equivalence':['CA','DB','EC','FD','GE',
                                                       'DA','EB','FC','GD','EA',
                                                       'FB','GC','FA','GB','GA']
                                        }
                        },

         "plot_blocks_ID": {
                            'nodal_distance':['Bsl','Sym','1-Tr','2-Tr','3-Tr',
                                              '4-Tr','5-Tr','1-Eq','2-Eq','3-Eq',
                                              '4-Eq','5-Eq'],
                            'relation_type':['Baseline','Symmetry',
                                            'Transivity','Equivalence']
                            },

        "mastery_training": 0.9
        },

# This is an alternative to the Sidman and Tailby experiment (1982)
    5: {
        "num_classes": 3,

        "training_order": {
                           1:[('A1','B1',10),('B2','A2',10)],
                           2:[('A1','B1',10),('A3','B3',10)],
                           3:[('B2','A2',10),('A3','B3',10)],
                           4:[('A1','B1',10),('B2','A2',10),('A3','B3',10)],
                           5:[('A1','C1',10),('C2','A2',10)],
                           6:[('A1','C1',10),('A3','C3',10)],
                           7:[('C2','A2',10),('A3','C3',10)],
                           8:[('A1','C1',10),('C2','A2',10),('A3','C3',10)],
                           9: [('A1','B1',5),('B2','A2',5),('A3','B3',5),
                               ('A1','C1',5),('C2','A2',5), ('A3','C3',5)],
                          10: [('D1','C1',10),('C2','D2',10)],
                          11:[('D1','C1',10),('D3','C3',10)],
                          12: [('C2','D2',10),('D3','C3',10)],
                          13:[('D1','C1',10),('C2','D2',10),('D3','C3',10)],
                          14:[('A1','B1',5),('B2','A2',5),('A3','B3',5),
                              ('A1','C1',5),('C2','A2',5), ('A3','C3',5),
                              ('D1','C1',5),('C2','D2',5),('D3','C3',5)]
                          },

        "plot_blocks": {
                        'relation_type':{
                                        'Baseline':['AB','AC','DC'],
                                        'Symmetry':['BA','CA','CD'],
                                        'Equivalence':['BC','CB','BD','DB','AD','DA']
                                        }
                        },

        "plot_blocks_ID": {'relation_type':['Baseline','Symmetry','Equivalence']},

        "mastery_training": 0.9
        },

# This is the Devany et. al. experiment (1986), by changing the training.
    6: {
        "num_classes": 2,

        "training_order": {
                           1:[('A1','B1',10)],
                           2:[('B2','A2',10)],
                           3:[('A1','B1',5),('B2','A2',5)],
                           4:[('A1','C1',10)],
                           5:[('C2','A2',10)],
                           6:[('A1','C1',5),('C2','A2',5)],
                           7: [('A1','B1',2),('B2','A2',2),('A1','C1',2),('C2','A2',2)]
                          },

        "plot_blocks": {},

        "plot_blocks_ID": {},

        "mastery_training": 0.9
        },

# This is OTM version of Spencer and Chase experiment (1996)
    7: { #
        "num_classes": 3,
        "training_order": {
                           1:[('A','B',48)],
                           2:[('A','B',24),('A','C',24)],
                           3:[('A','B',12),('A','C',12),('A','D',24)],
                           4:[('A','B',9),('A','C',9),('A','D',9),('A','E',24)],
                           5:[('A','B',6),('A','C',6),('A','D',6),('A','E',6),
                              ('A','F',24)],
                           6:[('A','B',3),('A','C',3),('A','D',3),('A','E',6),
                              ('A','F',9),('A','G',24)],
                           7:[('A','B',3),('A','C',3),('A','D',3),('A','E',3),
                              ('A','F',3),('A','G',3)]
                           },

        "plot_blocks": {
                        'relation_type':{
                                        'Baseline':['AB','AC','AD','AE','AF','AG'],
                                        'Symmetry':['BA','CA','DA','EA','FA','GA'],
                                        'Equivalence':['BC','BD','BE','BF','BG',
                                                       'CB','CD','CE','CF','CG',
                                                       'DB','DC','DE','DF','DG',
                                                       'EB','EC','ED','EF','EG',
                                                       'FB','FC','FD','FE','FG',
                                                       'GB','GC','GD','GE','GF']
                                        }
                        },

         "plot_blocks_ID": {
                            'relation_type':['Baseline','Symmetry','Equivalence']
                            },

        "mastery_training": 0.9
        },

# This is MTO version of Spencer and Chase experiment (1996)
    8: { #
        "num_classes": 3,
        "training_order": {
                           1:[('A','G',48)],
                           2:[('A','G',24),('B','G',24)],
                           3:[('A','G',12),('B','G',12),('C','G',24)],
                           4:[('A','G',9),('B','G',9),('C','G',9),('D','G',24)],
                           5:[('A','G',6),('B','G',6),('C','G',6),('D','G',6),
                              ('E','G',24)],
                           6:[('A','G',3),('B','G',3),('C','G',3),('D','G',6),
                              ('E','G',9),('F','G',24)],
                           7:[('A','G',3),('B','G',3),('C','G',3),('D','G',3),
                              ('E','G',3),('F','G',3)]
                           },

        "plot_blocks": {
                        'relation_type':{
                                        'Baseline':['AG','BG','CG','DG','EG','FG'],
                                        'Symmetry':['GA','GB','GC','GD','GE','GF'],
                                        'Equivalence':['AB','AC','AD','AE','AF',
                                                       'BA','BC','BD','BE','BF',
                                                       'CA','CB','CD','CE','CF',
                                                       'DA','DB','DC','DE','DF',
                                                       'EA','EB','EC','ED','EF',
                                                       'FA','FB','FC','FD','FE']
                                        }
                        },

         "plot_blocks_ID": {
                            'relation_type':['Baseline','Symmetry','Equivalence']
                            },

        "mastery_training": 0.9
        },

    # This is OTM version of Spencer and Chase experiment (1996)
    9: { #
        "num_classes": 3,
        "training_order": {
                           1:[('A','B',48)],
                           2:[('A','B',24),('C','A',24)],
                           3:[('A','B',12),('C','A',12),('A','D',24)],
                           4:[('A','B',9),('C','A',9),('A','D',9),('E','A',24)],
                           5:[('A','B',6),('C','A',6),('A','D',6),('E','A',6),
                              ('A','F',24)],
                           6:[('A','B',3),('C','A',3),('A','D',3),('E','A',6),
                              ('A','F',9),('G','A',24)],
                           7:[('A','B',3),('C','A',3),('A','D',3),('E','A',3),
                              ('A','F',3),('G','A',3)]
                           },

        "plot_blocks": {
                        'relation_type':{
                                        'Baseline':['AB','CA','AD','EA','AF','GA'],
                                        'Symmetry':['BA','AC','DA','AE','FA','AG'],
                                        'Equivalence':['BC','BD','BE','BF','BG',
                                                       'CB','CD','CE','CF','CG',
                                                       'DB','DC','DE','DF','DG',
                                                       'EB','EC','ED','EF','EG',
                                                       'FB','FC','FD','FE','FG',
                                                       'GB','GC','GD','GE','GF']
                                        }
                        },

         "plot_blocks_ID": {
                            'relation_type':['Baseline','Symmetry','Equivalence']
                            },

        "mastery_training": 0.9
        },


    11: { #
        "num_classes": 3,
        "training_order": {
                           1:[('A','B',30)],
                           2:[('B','C',30)],
                           3:[('C','D',30)],
                           4:[('D','E',30)],
                           5:[('E','F',30)],
                           6:[('F','G',30)],
                           },

        "plot_blocks": {},

         "plot_blocks_ID": {},

        "mastery_training": 0.9
        },

    12: { # OTM
        "num_classes": 6,
        "training_order": {
                           1:[('A','B',60)],
                           2:[('A','C',60)],
                           3:[('A','D',60)],
                           4:[('A','E',60)],
                           5:[('A','F',60)],
                           6:[('A','G',60)],
                           },

        "plot_blocks": {},

         "plot_blocks_ID": {},

        "mastery_training": 0.9
        },
     13: { # OTM
        "num_classes": 6,
        "training_order": {
                           1:[('A','G',60)],
                           2:[('B','G',60)],
                           3:[('C','G',60)],
                           4:[('D','G',60)],
                           5:[('E','G',60)],
                           6:[('F','G',60)],
                           },

        "plot_blocks": {},

         "plot_blocks_ID": {},

        "mastery_training": 0.9
        },

        }

    return environment_parameters_detail