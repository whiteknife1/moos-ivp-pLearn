from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Sequential, load_model
import numpy as np

from Constants import Constants
Constants = Constants()

class Agent():
    def __init__(self, actions):
        num_actions = len(actions)
        # enumerate state and action spaces
        self.index_by_action = {}
        for index, action in enumerate(actions):
            self.index_by_action[action] = index

        self.model_NN = self.create_model(num_actions)
        self.target_NN = self.create_model(num_actions)

    def create_model(self, num_actions):
        model = Sequential()
        for layer in range(Constants.num_layers):
            if layer == 0:
                nodes=Dense(units=Constants.num_units, input_dim=Constants.num_states+1, activation=Constants.activation_function)
            else:
                nodes=Dense(units=Constants.num_units,activation=Constants.activation_function)
            model.add(nodes)
            model.add(Dense(units=num_actions, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=Constants.lr), metrics=["accuracy"])
        return model


    def state2vec(self, s):
        temp=list(s)
        temp.append(1)
        for param in Constants.state:
            if Constants.state[param].standardized:
                if Constants.state[param].type != "binary":
                    temp[Constants.state[param].index]=float(temp[Constants.state[param].index]
                        -Constants.state[param].range[0])/Constants.state[param].range[1]
        return np.array([temp])

    def set_weights(self):
        self.target_NN.set_weights(self.model_NN.get_weights())

    def learn(self, cur_state, action, new_state, reward):
        cur_state = self.state2vec(cur_state)
        new_state = self.state2vec(new_state)
        baseline = self.model_NN.predict(cur_state)
        max_next = max(self.target_NN.predict(new_state)[0])
        baseline[0][self.index_by_action[action]] = reward+max_next*Constants.discount_factor
        self.model_NN.fit(cur_state, baseline, epochs=Constants.epochs)

    def save_model(self, save_dir):
        self.model_NN.save(save_dir+"model.h5")

    def load_models(self):
        self.model_NN.load_model(Constants.load_model_dir+"model.h5")
        self.target_NN.load_model(Constants.load_model_dir+"model.h5")
