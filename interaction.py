# -*- coding: utf-8 -*-
"""
Last update: Sep. 2, 2020

@author: Asieh Abolpour Mofrad

To mediate the interaction between agent and environment objects
based on the protocol and initializations.

This code is used for simulation results reported in an article entitled:
    ''Enhanced Equivalence Projective Simulation:
        a Framework for Modeling Formation of Stimulus Equivalence Classes"
        in Neural Computation, MIT Press.

"""

import sys
import copy
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

import pdb


class Interaction(object):

    """
    Interaction between agent and environment for Equivalence Projective Simulation
    """

    def __init__(self, agent, environment, agent_parameter, environment_parameter):

        """
        For a given agent, environment and their parameters, run the whole
        experiment and save the results under self.result_file_name.
        The results can be plotted using methods in the class
        """

        self.agent = copy.deepcopy(agent)
        self.environment = copy.deepcopy(environment)
        self.agent_parameter = agent_parameter
        self.environment_parameter = environment_parameter
        self.max_trial = environment_parameter['max_trial'][0]
        file_name = 'Env_'+ str(environment_parameter['environment_ID'][0])+ \
        '_ExpID_'+ str(environment_parameter['experiment_ID'][0])
        self.file_name = "results/{}.p".format(file_name)


    def run_experiment(self): # Ok!

        """ This method run the experiment for one agent/participant"""

        num_steps = 0
        while (self.environment.Training):
            if num_steps == self.max_trial:
                sys.exit("UNABLE TO FINISH TRAINING WITHIN {} STEPS".format(
                                                               self.max_trial))

            percept, action_set_t = self.environment.next_trial()
            num_steps += 1
            self.agent.trial_preprocess(percept, action_set_t)
            action = self.agent.action_selection(percept, action_set_t)
            reward = self.environment.feedback(percept, action)
            self.agent.training_update_network(percept, action_set_t,
                                               action, reward)


    def experiment_results(self): # Ok!

        """ This method run the experiment for num_agents and report the results"""

        num_agents = self.environment_parameter['num_agents'][0]

        avg_time_training = {}
        avg_prob_training = {}
        avg_NE_itr = 0

        agent = copy.deepcopy(self.agent)
        environment = copy.deepcopy(self.environment)

        for i_trial in range(num_agents):
            print(i_trial)
            self.environment = copy.deepcopy(environment)
            self.agent = copy.deepcopy(agent)
            self.run_experiment()

            if i_trial == 0:
                avg_time_training = self.environment.num_iteration_training.copy()
                avg_prob_training = self.environment.Block_results_training.copy()
                prob_training_clip = self.agent.softmax_matrix()
                W_in, P, Tau, prob_testing_clip = self.agent.Network_Enhancement()
                prob_testing_clip_marginalized = self.agent.marginalized_probability(prob_testing_clip)
                prob_testing_clip_category = self.agent.probability_categorization(prob_testing_clip_marginalized)
                avg_NE_itr += self.agent.NE_itr

            else:
                for k, v in self.environment.num_iteration_training.items():
                    avg_time_training[k] += v

                for k, v in self.environment.Block_results_training.items():
                    avg_prob_training[k] += v

                prob_training_clip += self.agent.softmax_matrix()
               # prob_testing_clip += self.agent.Network_Enhancement()
                W_in_, P_, Tau_, W_new_ = self.agent.Network_Enhancement()
                W_in += W_in_
                P += P_
                Tau += Tau_
                prob_testing_clip += W_new_
                prob_testing_clip_marginalized += self.agent.marginalized_probability(prob_testing_clip)
                prob_testing_clip_category += self.agent.probability_categorization(prob_testing_clip_marginalized)
                avg_NE_itr += self.agent.NE_itr

        for k, v in avg_time_training.items():
            avg_time_training[k] = v/ num_agents

        for k, v in avg_prob_training.items():
            avg_prob_training[k] = v/ num_agents

        prob_training_clip /= num_agents
        prob_testing_clip  /= num_agents
        prob_testing_clip_marginalized /= num_agents
        prob_testing_clip_category /= num_agents
        W_in /= num_agents
        P /= num_agents
        Tau /= num_agents

        training_df = self.training_dataframe(self.environment.training_order,
                                        avg_time_training, avg_prob_training)

        avg_NE_itr /= num_agents
        print('average number iteration', avg_NE_itr)

        results = [training_df, prob_training_clip, prob_testing_clip,
                   prob_testing_clip_marginalized, prob_testing_clip_category,
                   avg_NE_itr, W_in, P, Tau]

        return results


    def training_dataframe(self, training_order, avg_time_training,
                                                           avg_prob_training): # Ok!

        """
        To create a summery of training, including block size, average number of
        blocks and the final grade (mastery criteria).
        """

        train_list = []
        size_list = []
        time_list = []
        mastery_list = []
        for k, v in training_order.items():
            train = ''
            size=0
            for pair in v:
                train += pair[0]+pair[1]+', '
                size += pair[2]
            train_list.append(train)
            size_list.append(size)
            time_list.append(avg_time_training[k])
            mastery_list.append(avg_prob_training[k])
        df = pd.DataFrame({'Training': train_list,
                           'Block Size': size_list,
                            'Time': time_list,
                            'Mastery': mastery_list})
        return df


    def agent_results_avg(self, relations, prob_dict): #Ok!

        """
        This method recieves a set of relations or category pairs like
        {1:['A1B1', 'A2B2'], 2:['C1B1', 'C2B2']} or {1:['AB', 'BC'], 2:['CB', 'CD']}
        and a dictionary which the keys are like 'A1B1' or 'AB' and values are
        a number (probability)
        Output: a dictionary like {1:0.93, 2:0.78} where the numbers are the
        average values
        """

        avg_relations = {}
        for k, v in relations.items():
            sum_prob = 0
            i = 0
            for relation in v:
                sum_prob += prob_dict[relation]
                i += 1
            avg_relations[k] = sum_prob/i

        return avg_relations


    def df_to_dict(self, df): #Ok!

        """
        This method recieves a dataframe and return its dictionary format.
        """

        df_dict = {}
        for row in df.index:
            for col in df.columns:
                df_dict[row + col] = df.at[row, col]

        return df_dict


    def run_save(self): # Ok!

        """
        This is to save the results into pickle files for plotting
        and further calls.
        """

        results = self.experiment_results()
        show, result = self.plot_data(results)

        Simulation_data = {}
        Simulation_data['agent_parameter'] = self.agent_parameter
        Simulation_data['environment_parameter'] = self.environment_parameter
        Simulation_data['show'] = show
        Simulation_data['result'] = result

        result_save = open( self.file_name , "wb" )
        pickle.dump(Simulation_data, result_save)
        result_save.close()


    def plot_data(self, results): # Ok!

        """
        To save results for plot in the self.filaname address, and training
        results in a latex file.
        """

#        self.save_latex(results)

        training_df, prob_training_clip, prob_testing_clip, \
        prob_testing_clip_marginalized, prob_testing_clip_category, avg_NE_itr,\
        W_in, P, Tau = results

        plot_blocks = self.environment.plot_blocks
        plot_blocks_ID = self.environment.plot_blocks_ID

        show = []
        result = {}

        show.append(('W_in' , 'heatmap'))
        result['W_in'] = W_in

        show.append(('P matrix' , 'heatmap'))
        result['P matrix'] = P

        show.append(('Tau matrix' , 'heatmap'))
        result['Tau matrix'] = Tau

        result_1 = prob_testing_clip.copy()
        show.append(('General Pairwise probability', 'heatmap'))
        result['General Pairwise probability'] = result_1

        result_2 = prob_testing_clip_marginalized
        show.append(('Within category probability', 'heatmap'))
        result['Within category probability'] = result_2

        result_3 = prob_testing_clip_category
        show.append(('Category-to-category probability', 'heatmap'))
        result['Category-to-category probability'] = result_3

        test_prob_dict = self.df_to_dict(prob_testing_clip_category)
        index = sorted(test_prob_dict.keys())
        agn_1 = [test_prob_dict[k] for k in index]
        result_4 = pd.DataFrame({'Connection Probabilities': agn_1}, index = index)

        show.append(('Relation results', 'bar'))
        result['Relation results'] = result_4

        if bool(plot_blocks): # returns True if the dict is not empty
            for k_ , v_ in plot_blocks.items():
                agn_prob = self.agent_results_avg(v_, test_prob_dict)
                index = sorted(plot_blocks[k_].keys())
                index = plot_blocks_ID[k_]
                agn_2 = [agn_prob[k] for k in index]
                result_ = pd.DataFrame({'Connection Probabilities': agn_2},
                                        index = index)
                show.append((k_, 'bar'))
                result[k_] = result_

        return show, result


#    def save_latex(self, results):
#
#        training_df, prob_training_clip, prob_testing_clip, \
#        prob_testing_clip_marginalized, prob_testing_clip_category, avg_NE_itr,\
#        W_in, P, Tau = results
#
#        tf = open('latex_'+ self.file_name[:-1]+'txt', 'w')
#        tf.write(training_df.to_latex())
#
#        agent_ini = pd.DataFrame()
#        environment_ini = pd.DataFrame()
#        for k,v in self.agent_parameter.items():
#            if k == 'agent_type':
#                agent_ini[k] = [v]
#            else:
#                agent_ini[k] = [v[0]]
#        for k, v in self.environment_parameter.items():
#            environment_ini[k] = [v[0]]
#
#        tf.write(agent_ini.to_latex())
#        tf.write(environment_ini.to_latex())
#        tf.close()


class Plot_results(object):

    """
    This class plot the results which has been stored in 'file name' location.
    """

    def __init__(self, file_name):

        self.file_name=file_name

    def showResults(self):

        """
        reads the data and desired representations and plot them,
        an alternative to plot_results method
        """

        resultFile = open(self.file_name, 'rb')
        data = pickle.load(resultFile)
        resultFile.close()
        for i in range(len(data['show'])):
            result = data['result'][data['show'][i][0]]
            showType = data['show'][i][1]
            name = data['show'][i][0] + '_' + showType

            # show the result
            if showType == 'bar':
                self.barDiagramShow(name, result)    # result is a dataframe
            elif showType == 'heatmap':
                self.heatmapShow(name, result)       # result is a dataframe


    def barDiagramShow(self, name,  data):

        """ bar plot"""

        data.plot(kind = 'bar', color = ['royalblue','lightgreen', 'red','cyan'])
        plt.legend(fontsize = 20)
        plt.tick_params(labelsize = 20)
        plt.title(name, fontsize = 20)
        plt.ylabel('Correct match ratio', fontsize = 20)
        plt.tight_layout()
        plt.xticks( rotation=45, fontsize = 18, horizontalalignment = 'right')
        plt.show()


    def heatmapShow(self, name, data):

        """ heatmap plot"""

        fig, ax = plt.subplots()
      #  sns.set(font_scale = 1.5)
        sns.heatmap(data.round(3),xticklabels=True, yticklabels=True, annot = True,
                    annot_kws = {"size": 14}, linewidths =.15, fmt="g", cmap="Blues") # cmap="Greens"

        plt.title(name, fontsize = 16)
        plt.tick_params(labelsize = 16)
        plt.tight_layout()
        plt.show()

    def print_setting(self):

        """ for printing the setting of the simulation in the consule"""

        resultFile = open(self.file_name, 'rb')
        data = pickle.load(resultFile)
        resultFile.close()

        agent_parameter = data['agent_parameter']
        environment_parameter = data['environment_parameter']

        agent_ini = pd.DataFrame()
        environment_ini = pd.DataFrame()
        for k, v in agent_parameter.items():
            if k == 'agent_type':
                agent_ini[k] = [v]
            else:
                agent_ini[k] = [v[0]]
        for k, v in environment_parameter.items():
            environment_ini[k] = [v[0]]

        print("---*Agent Setting*---", agent_ini, sep = '\n')
        print("\n---*Environment Setting*---", environment_ini, sep = '\n')
