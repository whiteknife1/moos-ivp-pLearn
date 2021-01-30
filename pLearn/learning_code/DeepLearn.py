
"""
Created on Mon Nov 20 10:24:39 2017

@author: Arjun Gupta
"""

import pdb
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import random
from collections import deque
from keras.models import Sequential, Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Dense, Flatten
from keras.layers.merge import Add
from keras.optimizers import Adam
from keras import optimizers
import sys
import time
import os.path
import tensorflow as tf


from DQL import Agent
#import constants to be used
from Constants import Constants

Constants=Constants()
random.seed(time.clock())


class Deep_Learner:
    """
    Initializes a DeepLearner Instance, and sets up basic instance variables
    """
    def __init__(self,reward_fn, actions, alg_type, sess, load=False):
        #dictionary for Q-Value pairs
        self.q = {}

        #enumerated state and action spaces
        self.actions = actions
        self.index_by_action = {}
        for index, action in enumerate(self.actions):
            self.index_by_action[action] = index

        #reward_fn(state) returns the reward for that state (in this implementation, it is independent of action)
        self.reward_fn = reward_fn

        #decide whether to load previously learned weights or not
        self.load = load

        #copy in terms from Constants.py
        self.num_layers = Constants.num_layers
        self.num_units = Constants.num_units
        self.num_traj = Constants.num_traj
        self.iters = Constants.iters
        self.lr = Constants.lr
        self.eps_min = Constants.eps_min
        self.eps_decay = Constants.eps_decay
        self.eps = Constants.eps_init
        self.batch_size = Constants.batch_size
        self.epochs = Constants.epochs
        self.discount_factor = Constants.discount_factor
        self.iteration = 0
        self.alg_type = alg_type
        self.sess = sess
        self.initialize()

    """
    Initializes a Deep_Learner instance that can train a model, and initializes
    the model specific fields
    """
    def initialize(self):
         #initialize memory bank of state transitions and rewards
        if(Constants.mem_type=="set"):
             #use a set so that all (s, a, s', r) tuples have equal chance of being chosen
            self.memory = set()
        elif(Constants.mem_type == "deque"):
            #use deque to make experience we have more often have a higher chance of being picked
            self.memory = deque(maxlen=Constants.mem_length)
        elif(Constants.mem_type =="memory per action"):
            self.memory = {}
            for action in self.actions:
                self.memory[action] = deque(maxlen=Constants.mem_length)
        #bad memory to emphasize negative occurences to train on
        self.bad_memory = deque(maxlen=Constants.mem_length)

        #output the environment into the folder
        subprocess.call(["mkdir",Constants.save_model_dir])
        self.output_environment()

        #initalize folder at 0 iterations if it doesn't already exist
        if(not os.path.isfile(Constants.save_model_dir+"iterations")):
            with open(Constants.save_model_dir+"iterations", "w") as file:
                file.write('0')
        else:
            with open(Constants.save_model_dir+"iterations", "r") as file:
                self.iteration = int(file.read())//Constants.save_iter_num

        #load in memory from previous iterations
        self.load_memory()

        #set configure the model save directory
        if Constants.save_iteration:
            self.save_dir=Constants.save_model_dir+"/iteration_"+str(self.iteration)+"/"
        else:
            self.save_dir=Constants.save_model_dir

        #updates iteration file and save_dir if Constants.save_iteration
        self.iteration=self.update_iters()

        print("saving model to "+self.save_dir)

        #initialize models
        if self.alg_type == "DQL":
            self.dql_agent = Agent(self.actions)
            if self.load:
                self.dql_agent.load_models()
            self.dql_agent.save_model(self.save_dir)
        else:
            raise SyntaxError("alg_type: "+self.alg_type+" not supported. Please specify either 'fitted', 'DQL', or 'A/C' as your alg_type.")


    """
    Procedure: sim_episode()

      Purpose: Runs a simulation for trial_num iterations, and saves data into memory to be used later

      Returns: The average reward obtained over all iterations
    """
    def sim_episode(self, trial_num, iters=Constants.num_traj, read_path=Constants.read_path,
                    process_cmd=Constants.process_cmd, process_path=Constants.process_path,
                    simulation_cmd=Constants.sim_cmd, table = True):
        """
        function to interface with the simulation_engine (moos simulator). Calls subfunction
        that runs and processes moos_data
        """

        #output action table for behavior to read
        if(table):
            self.output_table()

        #set up trajectory and open a file to write the training data to for checking/reloading later
        traj=[]
        total_trials = 0
        trials_out = 0
        total_reward = 0
        flag_captured = 0
        '''Lets create directories '''
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('processed'):
            os.makedirs('processed')
        if not os.path.exists("paths"):
            os.makedirs("paths")
        write_file=open('paths/trajectory_'+str(trial_num)+'.log', 'w')
        for _ in range(iters):
            trial_num+=1
            #call simulation
            subprocess.call([simulation_cmd, str(trial_num)])

            #process simulation.alog with log_converter.py
            process_file=process_path+'/simulation_'+str(trial_num)+'/simulation.alog'
            read_file=read_path+'/simulation_'+str(trial_num)+'.alog'
            cmd=['python',process_cmd,process_file,read_file]
            print("processing data from", process_file, "using", process_cmd,"to", read_file)
            subprocess.call(cmd)

            #read in processed data -- form: state action next_state
            with open(read_file, 'r') as inp:
                #split into lines and store in array
                data=inp.read().splitlines()

            captured = False
            for line in data:
                total_trials+=1
                #construct (s_0, a, s_1, r) lists for consumption later
                terms=line.split(' ')
                add_list=[]

                for term in terms:
                    tuple_term=[]

                    for num in term.split(','):
                        try:
                            tuple_term.append(int(num))
                        except:
                            tuple_term.append(float(num))
                    add_list.append(tuple(tuple_term))

                reward=self.reward_fn(add_list[2])
                add_list.append(reward)
                total_reward += reward
                # print("Total Reward: "+str(total_reward))

                if(add_list[0][Constants.state["flag_dist"].index] <= 10):
                    captured = True

                if(add_list[0][Constants.state["out"].index]==0):
                    trials_out+=1

                if add_list[1] != (0,0) and  add_list[0] != add_list[2]:
                    #only add it if it is a valid action
                    if(Constants.end_at_tagged):
                        if(add_list[0][Constants.state["out"].index]==0):
                            traj.append(add_list)
                            if(Constants.mem_type == "set"):
                                self.memory.add(tuple(add_list))
                            elif(Constants.mem_type == "deque"):
                                self.memory.append(tuple(add_list))
                            else:
                                self.memory[add_list[1]].append(tuple(add_list))
                            if(add_list[2][Constants.state["out"].index]==1):
                                self.bad_memory.append(tuple(add_list))
                                print("added "+str(add_list)+" to bad memory")
                    else:
                        traj.append(add_list)
                        if(Constants.mem_type == "set"):
                            self.memory.add(tuple(add_list))
                        elif(Constants.mem_type == "deque"):
                            self.memory.append(tuple(add_list))
                        else:
                            self.memory[add_list[1]].append(tuple(add_list))

                write_file.write(str(add_list)+"\n")
            flag_captured+= int(captured)

        if(Constants.mem_type == "memory per action"):
            print("Memory sizes")
            for action in self.actions:
                print(len(self.memory[action]))

        if(total_trials != 0):
            time_out = float(trials_out)/total_trials
            pct_captured = float(flag_captured)/trial_num
        else:
            time_out = 0
            pct_captured = 0

        write_file.close()
        return (time_out, total_reward, pct_captured)

    """
    Procedure: save_memory()

      Purpose: saves s_0, a, s_1, r tuples to file to be used in later training sets

      Returns: void
    """
    def save_memory(self):
        print("saving experiences...")
        with open(Constants.save_model_dir+"memory.txt", 'w') as mem:
            if(Constants.mem_type == "A/C"):
                for m in self.memory:
                    mem.write(str(m)+"\n")
            elif(Constants.mem_type != "memory per action"):
                for thing in self.memory:
                    mem.write(str(thing)+" \n")
            else:
                for action in self.memory.keys():
                    for memory in self.memory[action]:
                        mem.write(str(memory)+" \n")
        if(Constants.end_at_tagged):
            with open(Constants.save_model_dir+"bad_memory.txt", 'w') as mem:
                for thing in self.bad_memory:
                    mem.write(str(thing)+"\n")

    """
    Procedure: load_memory()

      Purpose: loads s_0, a, s_1, r, tuples from file into memory for training

      Returns: void
    """
    def load_memory(self):

        if(Constants.mem_address != "" and Constants.mem_type != "memory per action"):
            for mem in ["memory.txt", "bad_memory.txt"]:
                try:
                    with open(Constants.mem_address+mem, 'r') as infile:
                        #read in data
                        data = infile.read().splitlines()
                        for line in data:
                            term = []
                            last_pointer = 0
                            #seperate line into s_0, a, s_1, rand put into list
                            for i, character in enumerate(line):
                                if character == '(':
                                    last_pointer = i
                                elif character == ')':
                                    term.append(line[last_pointer+1:i])
                                    last_pointer = i+1

                                #convert to numerical values and restore as tuples
                            for i, value in enumerate(term):
                                sub_list=[]
                                for num in value.split(','):
                                    try:
                                        temp = float(num)
                                    except:
                                        temp = int(num)
                                    sub_list.append(temp)
                                term[i] = tuple(sub_list)
                            #reformat last term (reward) so that it is just the number
                            term[-1] = term[-1][0]
                            #store line into memory
                            if(mem == "memory.txt"):
                                if(Constants.mem_type=="set"):
                                    self.memory.add(tuple(term))
                                else:
                                    self.memory.append(tuple(term))
                            else:
                                self.bad_memory.append(tuple(term))
                except:
                    pass
            print("loaded "+str(len(self.memory))+" experiences into memory")
            print("loaded "+str(len(self.bad_memory))+" experiences into bad_memory")



    """
    Procedure: output_table()

      Purpose: Outputs the location of the most recent model Neural Net as well as all important
               fields to configure BHV_Input

      Returns: void
    """
    def output_table(self, out_address=Constants.out_address, model_address="",  optimal=False):
        print("outputting table")
        if(model_address == ""):
            model_address = self.save_dir

        #update epsilon and report value
        if not optimal:
            self.eps *= self.eps_decay
            self.eps = max(self.eps_min, self.eps)
            print("Epsilon: "+ str(self.eps))
        else:
            print("Epsilon: optimal")

        with open(out_address, 'w') as file:
            #write important state information for BHV_Input to consume
            file.write("relative="+str(Constants.relative)+"\n")
            file.write("num_states="+str(Constants.num_states)+"\n")
            file.write("players="+Constants.players+"\n")
            file.write("state=")
            for i in range(len(Constants.state)):
                key=Constants.state.keys()[i]
                file.write(str(key))
                file.write(" type+"+str(Constants.state[key].type))
                file.write(" bucket+"+str(Constants.state[key].bucket))
                file.write(" var+"+str(Constants.state[key].var))
                file.write(" var_mod+"+str(Constants.state[key].var_mod))
                file.write(" index+"+str(Constants.state[key].index))
                file.write(" vehicle+"+str(Constants.state[key].vehicle))
                if i != len(Constants.state)-1:
                    file.write(",")
            file.write("\n")
            file.write("model_address="+model_address+"\n")
            file.write("actions=")
            for i, action in enumerate(self.actions):
                print(str(action))
                file.write(str(action))
                if(i != len(self.actions)-1):
                    file.write(":")
            file.write("\n")
            file.write("optimal="+str(optimal)+"\n")
            if(not optimal):
                file.write("epsilon="+str(self.eps)+"\n")
            else:
                file.write("epsilon="+str(0)+"\n")

##################################################################################################################
    # Code for Deep Q Learning:                                                                                      #
    #     Uses One Neural net which takes in the state as input and ouputs all the Q-values for each action possible #
    #     Given that state                                                                                           #
    # Based very loosely on a tutorial found at:                                                                     #
    # https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c                          #
    ##################################################################################################################

    """
    Procedure: DQL()

      Purpose: runs the main loop for running simulations, and training the Deep Q Neural Net

      Returns: void
    """
    def DQL(self):
        for iter in range(self.iteration, self.iters):
            print("---------------------- On iteration "+ str(iter+1)+ " of "+ str(self.iters)+" -----------------------")
            trial=iter*self.num_traj
            #simulates episode and puts relevant experiences into memory bank
            self.sim_episode(trial_num=trial)
            self.DQL_train_model()
            #update target network
            self.dql_agent.set_weights()
            self.save_memory()
            if Constants.save_iteration:
                self.update_iters()
            self.dql_agent.save_model(self.save_dir)
    """
    Procedure: DQL_train_model()

      Purpose: trains the Deep Q Learner by selecting a random sample from memory and training on it

      Returns: void
    """
    def DQL_train_model(self):
         #randomly select batch of simulations from memory bank
        if self.batch_size > len(self.memory) and Constants.mem_type != "memory per action":
            return
        else:
            #select memories evenly for every action
            if(Constants.mem_type == "memory per action"):
                data = []
                for action in self.actions:
                    if(len(self.memory[action]) >  self.batch_size//(len(self.actions))):
                       subdata = random.sample(self.memory[action], self.batch_size//(len(self.actions)))
                       data.extend(subdata)
                    else:
                       data.extend(self.memory[action])
            else:
                data = random.sample(self.memory, self.batch_size)

            #make predictions with target network, train neural network based on target network prediction for "batch_size" iterations
            for cur_state, action, new_state, reward in data:
                self.dql_agent.learn(cur_state, action, new_state, reward)


    #################################
    # Helper functions
    #################################

    """
    Procedure: epsilon_greedy()
      Purpose: Selects a random action with probability epsilon, and otherwise returns the action
               that has maximum predicted Q_value base on the neural net
      Returns: epsilon-optimal action for the current state
    """
    def epsilon_greedy(self, state):
        if np.random.random_sample() < self.eps:
            return random.sample(self.actions, 1)[0]
        return self.optimal_action(state)

    """
    Procedure: optimal_action()
      Purpose: returns the action that has maximum predicted Q_value base on the neural net
      Returns: optimal action for the current state
    """
    def optimal_action(self, s):
        if(self.alg_type == "fitted"):
            opt=(0,None)
            for action in self.actions:
                act_val=self.approx_q_value(s, action)
                if act_val >= opt[0] or opt[1]==None:
                    opt=(act_val, action)
            return opt[1]
        else:
            opt = self.actions[np.argmax(self.model_NN.predict(self.state2vec(s))[0])]
            return opt
    """
    Procedure: state2vec()
      Purpose: Converts the state into a numpy array to be passed through the neural net
      Returns: the state vector in the form of a numpy array
    """

    def state2vec(self, s):
        temp=list(s)
        temp.append(1)
        for param in Constants.state:
            if Constants.state[param].standardized:
                if Constants.state[param].type != "binary":
                    temp[Constants.state[param].index]=float(temp[Constants.state[param].index]
                        -Constants.state[param].range[0])/Constants.state[param].range[1]
        return np.array([temp])

    """
    Procedure: output_environment()

      Purpose: helper function to output all the important information in Constants.py when the program is run

      Returns: void
    """
    def output_environment(self):
        subprocess.call(["cp", "Constants.py", Constants.save_model_dir+"environment.py"])
        with open(Constants.save_model_dir+"environment.txt", "w") as out_file:
            out_file.write("State Definition: \n \n")
            for param in Constants.state:
                out_file.write("   "+param+": index= "+str(Constants.state[param].index))
                out_file.write(", type= "+Constants.state[param].type)
                out_file.write(", var= "+ Constants.state[param].var)
                out_file.write(", var_mod= "+Constants.state[param].var_mod)
                if(Constants.state[param].type != "binary"):
                    out_file.write(", standardized= "+str(Constants.state[param].standardized))
                    out_file.write(", range= "+str(Constants.state[param].range))
                    out_file.write(", bucket= "+str(Constants.state[param].bucket))
                out_file.write("\n")
            out_file.write("\n")

            out_file.write("Neural Net Parameters: \n \n")
            out_file.write("   "+"num_layers= "+str(Constants.num_layers)+"\n")
            out_file.write("   num_units= "+str(Constants.num_units)+"\n")
            out_file.write("   num_traj= "+str(Constants.num_traj)+"\n")
            out_file.write("   iterations= "+str(Constants.iters)+"\n")
            out_file.write("   learning_rate= "+str(Constants.lr)+"\n")
            out_file.write("   epsilon_min= "+str(Constants.eps_min)+"\n")
            out_file.write("   epsilon_initial= "+str(Constants.eps_init)+"\n")
            out_file.write("   epsilon_decay= "+str(Constants.eps_decay)+"\n")
            out_file.write("   epochs= "+str(Constants.epochs)+"\n")
            out_file.write("   batch_size= "+str(Constants.epochs)+"\n")
            out_file.write("   algorithm_type= "+str(Constants.alg_type)+"\n \n")

            out_file.write("Action and Reward Parameters: \n \n")
            out_file.write("   speeds= "+str(Constants.speeds)+ "\n")
            out_file.write("   relative= "+str(Constants.relative)+"\n")
            if(Constants.relative):
                out_file.write("   relative_headings= "+str(Constants.rel_headings)+"\n")
            out_file.write("   theta_size_act= "+str(Constants.theta_size_act)+"\n")
            out_file.write("   disctount_factor= "+str(Constants.discount_factor)+"\n")
            out_file.write("   max_reward= "+str(Constants.max_reward)+"\n")
            out_file.write("   neg_reward= "+str(Constants.neg_reward)+"\n")
            out_file.write("   reward_dropoff= "+str(Constants.reward_dropoff)+"\n")
            out_file.write("   max_reward_radius= "+str(Constants.max_reward_radius)+"\n")

    """
    Procedure: update_iters()

      Purpose: increments the current iteration number and iteration file by 1

      Returns: the current iteration number
    """
    def update_iters(self):
        """
        helper function to update the iterations file and change the save directory
        """
        with open(Constants.save_model_dir+"iterations", "r") as file:
            iterations=int(file.read())

        with open(Constants.save_model_dir+"iterations", "w") as file:
            file.write(str(iterations+1))

        if iterations%Constants.save_iter_num==0:
            self.save_dir=Constants.save_model_dir+"iteration_"+str(iterations//Constants.save_iter_num)+"/"
            subprocess.call(["mkdir", self.save_dir])
            #subprocess.call(["mkdir", self.save_dir+"weights/"])

        return iterations


