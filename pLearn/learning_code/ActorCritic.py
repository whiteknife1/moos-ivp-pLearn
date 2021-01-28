from keras.layers.core import Dense
from keras.optimizers import Adam
from tensorflow.distributions import Categorical
from tensorflow import GradientTape
from tensorflow.keras import Model
from tensorflow import saved_model
import tensorflow as tf
import numpy as np

from Constants import Constants
Constants = Constants()

class Actor(Model):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.dense_1 = Dense(Constants.num_units, activation=Constants.activation_function)
        self.dense_2 = Dense(Constants.num_units*2, activation=Constants.activation_function)
        self.dense_3 = Dense(Constants.num_units, activation=Constants.activation_function)
        self.actions = Dense(num_actions, activation='softmax')

    def call(self, input_data):
        x = self.dense_1(input_data)
        x = self.dense_2(x)
        x = self.dense_3(x)
        actions = self.actions(x)
        return actions

class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense_1 = Dense(Constants.num_units, activation=Constants.activation_function)
        self.dense_2 = Dense(Constants.num_units*2, activation=Constants.activation_function)
        self.dense_3 = Dense(Constants.num_units, activation=Constants.activation_function)
        self.value = Dense(1, activation=None)

    def call(self, input_data):
        x = self.dense_1(input_data)
        x = self.dense_2(x)
        x = self.dense_3(x)
        value = self.value(x)
        return value

class Agent():
    def __init__(self, num_actions, gamma = Constants.discount_factor):
        self.actor_opt = Adam(lr = Constants.lr)
        self.critic_opt = Adam(lr = Constants.lr)
        self.actor = Actor(num_actions)
        self.critic = Critic()

    def state2vec(self, s):
        temp=list(s)
        temp.append(1)
        for param in Constants.state:
            if Constants.state[param].standardized:
                if Constants.state[param].type != "binary":
                    temp[Constants.state[param].index]=float(temp[Constants.state[param].index]
                        -Constants.state[param].range[0])/Constants.state[param].range[1]
        return np.array([temp])

    def act(self, state):
        prob = self.actor(self.state2vec(state))
        dist = Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return self.state2vec(action)

    def actor_loss(self, prob, action, temporal_diff):
        dist = Categorical(probs=prob, dytpe=dtype.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*temporal_diff
        return loss

    def learn(self, cur_state, action, new_state, reward):
        cur_state = self.state2vec(cur_state)
        new_state = self.state2vec(new_state)
        with GradientTape() as actor_tape, GradientTape() as critic_tape:
            prob = self.actor(state, training=True)
            value = self.critic(state, training=True)
            value_new = self.critic(new_state, training=True)
            temporal_diff = reward+self.gamma*value_new-value
            actor_loss = self.actor_loss(prob, action, temporal_diff)
            critic_loss = temporal_diff**2
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        return actor_loss, critic_loss

    def save_agent(self, save_dir):
        saved_model.save(self.actor, save_dir+"actor")
        saved_model.save(self.critic, save_dir+"critic")

    def load_agent(self):
        self.actor = saved_model.load(Constants.load_model_dir+"actor")
        self.critic = saved_model.load(Constants.load_model_dir+"critic")
