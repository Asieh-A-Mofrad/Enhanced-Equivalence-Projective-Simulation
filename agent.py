# -*- coding: utf-8 -*-
"""
Last update: Sep. 2, 2020

@author: Asieh Abolpour Mofrad

This code is used for simulation results reported in an article entitled:
    ''Enhanced Equivalence Projective Simulation:
        a Framework for Modeling Formation of Stimulus Equivalence Classes"
        in Neural Computation, MIT Press.

"""

import numpy as np
import networkx as nx
import pandas as pd
import sympy as sp

sp.init_printing(use_unicode=True)


class Agent(object):

    """
    Projective Simulation agent for Equivalence Class formation.
    """

    def __init__(self, parameter):

        """Initialize the basic PS agent,
        """

        self.gamma_damping = parameter["gamma_damping"][0]
        self.beta_h = parameter["beta_h"][0]
        self.beta_t = parameter["beta_t"][0]
        self.K = parameter["K"][0]
        self.alpha = parameter["alpha"][0]
        self.NE = parameter["network_enhancement"][0]
        self.NE_itr = 0
        self.clip_space = nx.DiGraph()


    def trial_preprocess(self, percept, action): # Ok!

            """
            Takes a percept and an action set, updates clip_space if required.
            """

            for act in  action:
                if (percept, act) not in self.clip_space.edges():
                    self.clip_space.add_edge(percept, act, weight=1)
                    self.clip_space.add_edge(act, percept, weight=1)


    def action_selection(self, percept, action_set_t, clip = None): # Ok!

        """Given a percept and an action set, this method returns the next action

        Arguments:
            - percept: any immutable object (as specified for trial_preprocess),
            - action_set_t: a list of any immutable object (as specified for
                                                            trial_preprocess),
        Output: action"""

        if clip is None:
            clip = self.clip_space.copy()

        h_vector = [clip[percept][action]['weight'] for action in action_set_t]
        probability_distr = self.softmax(h_vector, self.beta_h)
        size_action_set = len(action_set_t)
        Action = np.random.choice(size_action_set, p = probability_distr)

        return action_set_t[Action]


    def training_update_network(self, percept, action_set_t, action, reward): # Ok!

        """Given a history of what happend, i.e. the percept, the action set,
        the chosen action by the agent
        and the reward, this method updates the clip_space,
        the method is for the case where h_valuse could get negative values,

        Arguments:

            - percept: any immutable object (as specified for trial_preprocess),
            - action_set_t: a list of any immutable object (as specified for
                                                            trial_preprocess),
            - action: what agent chose for the above percept and action_set_t
            - reward: 1 or -1
        """

        for u, v, d in self.clip_space.edges(data=True):
            d['weight']= (1- self.gamma_damping) *d['weight']

        self.clip_space[percept][action]['weight'] += reward
        self.clip_space[action][percept]['weight'] += (self.K * reward)


    def softmax(self, h_vec, beta): # Ok!

        """Compute softmax values for each sets of h-values in h_vec."""

        h = [i* beta for i in h_vec]
        e_h = np.exp(h - np.max(h))
        prob = e_h / np.sum(e_h)

        return prob


    def softmax_revers(self, prob, beta): # Ok!
        """
        Compute h-values from a probability distribution vector. The h_vec is
        a positive vector with minimum value 1.
        """

        h = [i/ beta for i in np.log(prob)]
        h_vec = h - np.min(h) + 1

        return h_vec


    def category_list(self, Tr_martix):# Ok!

        """ To find the set of categories from the Transition matrix of the network"""

        category=set()
        for node in Tr_martix.index:
            category |= set(node[0])
        return sorted(list(category))


    def Network_Enhancement(self, W_in = None): # Ok!

        """
        W_in as the input to the network
        is either given or will be obtain from self.clip_space
        """

        if W_in is None:
            clip = self.clip_space.copy()
            nlist = sorted(clip.nodes())
            W_in = nx.to_pandas_adjacency(clip, nodelist = nlist)
            for i in W_in.index:
                W_in.at[i, i] = W_in.max(axis = 1)[i]

        P = self.softmax_matrix(W_in)
        W_old = P.copy()

        if self.NE:
            Tau = self.Tau_matrix(P, W_old.copy())
        else:
            Tau = P.copy()

        Error = 10
        self.NE_itr = 0
        while Error > 0.0001:
            W_new = self.alpha*Tau.dot(W_old).dot(Tau) + (1-self.alpha)*Tau
            W_error = W_new.copy()
            Error = W_error.sub(W_old).abs().sum().sum()
            W_old = W_new.copy()
            self.NE_itr += 1

#        clip_infty = self.theoretical_convergence(Tau)
#        W_error2 = W_new.copy()
#        Error_2 = W_error2.sub(clip_infty).abs().sum().sum()
#        print (np.allclose(W_new, clip_infty))
#        if Error_2 > 0.001:
#            print('Error: ', Error_2)

        return W_in, P, Tau, W_new


    def Tau_matrix(self, P, W): # Ok!

        """
        This methods takes P matrix as input and returns the Tau matrix.
        """

        Tau = P.copy()
        Tau[:] = 0
        for i in P.index:
            for j in P.columns:
               Tau.at[i,j] = np.sum([(P.at[i,k]*P.at[j,k])/P[k].sum()
                                                      for k in P.columns])

        return Tau


    def softmax_matrix(self, Tr_matrix = None, beta = None): # Ok!

        """Compute softmax values for each row of the matrix."""

        if Tr_matrix is None:
            clip = self.clip_space.copy()
            nlist = sorted(clip.nodes())
            Tr_matrix = nx.to_pandas_adjacency(clip, nodelist = nlist)

        if beta is None:
            beta = self.beta_h

        prob_matrix = Tr_matrix.copy()
        for i in Tr_matrix.index:
            h = beta * Tr_matrix.loc[i,:]
            e_h = np.exp(h - np.max(h))

            for j in range(len(h)):
                if h[j] == 0:
                    e_h[j] = 0
            prob_matrix.loc[i, :] = e_h/ np.sum(e_h)

        return prob_matrix


    def marginalized_probability(self, Tr_matrix): # Ok!

        """Compute probability distributions for each category of the matrix."""

        prob_matrix = Tr_matrix.copy()
        category = self.category_list(Tr_matrix)
        for row in Tr_matrix.index:
            for ctg in category:
                h_indx = [col for col in Tr_matrix.columns if col[0] == ctg]
                h = [Tr_matrix.loc[row, col] for col in h_indx]
                if np.sum(h) != 0:
                    h = h / np.sum(h)
                h_prob = self.softmax(h, self.beta_t)
                k = 0
                for col in h_indx:
                    prob_matrix.loc[row, col] = h_prob[k]
                    k += 1

        return prob_matrix


    def probability_categorization(self, prob_matrix): # Ok!

        """
        This method recieves a dataframe with categorized probabilities and returns
        a new table that indexes and columns are categories say, 'A', 'B', with probabilities.
        Also return a table with the same index but marginalized probabilities.
        """

        feature_list = self.category_list(prob_matrix)
        category_matrix = pd.DataFrame(0, index = feature_list, columns = feature_list)

        for ctg1 in feature_list:
            for ctg2 in feature_list:
                pr_sum_correct = 0
                pr_sum_wrong = 0
                for row in prob_matrix.index:
                    if row[0] == ctg1[0]:
                        pr_row_correct = [prob_matrix.at[row, col] for col in
                                          prob_matrix.columns if
                                          (col[0] == ctg2[0] and row[1] == col[1])]
                        pr_row_wrong = [prob_matrix.at[row, col] for col in
                                          prob_matrix.columns if (col[0] == ctg2[0]
                                          and row[1] != col[1])]
                        pr_sum_correct += np.sum(pr_row_correct)
                        pr_sum_wrong += np.sum(pr_row_wrong)
                ctg_pr = pr_sum_correct/ (pr_sum_correct+ pr_sum_wrong)
                category_matrix.loc[ctg1, ctg2] = ctg_pr

        return category_matrix


    def theoretical_convergence(self, Tau): #Ok!

        """
        To calculate the theorethical converged equilibrium graph using P
        """

        I =  Tau.copy()
        I[:] = 0
        for i in Tau.index:
            I.at[i,i] = 1

        W_infty = (1-self.alpha)*Tau.dot(np.linalg.inv(I.sub(self.alpha*Tau.dot(Tau))))
        W_infty.columns = Tau.columns

        return W_infty


    def class_based_sort(self, W): #Ok!

        """
        To re-order the matrix based on classes
        """

        new_list = [i[1] for i in W.index]
        W['class'] = new_list
        group_df = []
        for group, frame in W.groupby('class'):
            group_df.append(frame)

        W_class = pd.concat(group_df)
        cols = W_class.index
        W_class = W_class[cols]

        return W_class
