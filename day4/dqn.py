import tensorflow as tf
import numpy as np
from collections import deque
import random

NUM_CHANNELS = 4 # image channels
IMAGE_SIZE = 84  # 84x84 pixel images
SEED = None # random initialization seed
STATE_SIZE = 4
NUM_ACTIONS = 2  # number of actions for this game
BATCH_SIZE = 100
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
GAMMA = 0.9
RMS_LEARNING_RATE = 0.00025
RMS_DECAY = 0.99
RMS_MOMENTUM = 0.0
RMS_EPSILON = 1e-6
MAX_REPLAY_MEMORY = 10000

def weight_variable(shape, sdev=0.1):
    initial = tf.truncated_normal(shape, stddev=sdev, seed=SEED)
    return tf.Variable(initial)

def bias_variable(shape, constant=0.1):
   initial = tf.constant(constant, shape=shape)
   return tf.Variable(initial)

class QNet:
    def __init__(self, num_actions):
        # TODO:add a fully  connected layer with 16 neurons using weight and bias variables defined above

        # TODO: add another fully connected layer with num_actions neurons to represent the output.
        self.fc_w = ???
        self.fc_b = ???
        self.num_actions = num_actions
        self.out_w = ???
        self.out_b = ???

        self.stateInput = tf.placeholder("float", [None, STATE_SIZE])

        stateDims = np.prod(self.stateInput.get_shape().as_list()[1:])
        reshape = tf.reshape(self.stateInput, [-1, stateDims])
        hidden = ???
        # calculate the Q value as output
        self.QValue = ???

    def properties(self):
        return (self.fc_w, self.fc_b, self.out_w, self.out_b)

class DQN:
    def __init__(self, actions):
        self.replayMemory = deque()
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        self.currentQNet = QNet(len(actions))
        self.targetQNet = QNet(len(actions))

        self.actionInput = tf.placeholder("float", [None, len(actions)])
        self.yInput = tf.placeholder("float", [None])
        self.Q_action = tf.reduce_sum(tf.mul(self.currentQNet.QValue, self.actionInput), reduction_indices=1)
        tf.scalar_summary("q value", self.Q_action)
        self.loss = tf.reduce_mean(tf.square(self.yInput - self.Q_action))
        tf.scalar_summary("loss", self.loss)
        self.trainStep = tf.train.RMSPropOptimizer(RMS_LEARNING_RATE, RMS_DECAY, RMS_MOMENTUM, RMS_EPSILON).minimize(self.loss)

    def copyCurrentToTargetOperation(self):
        targetProps = self.targetQNet.properties()
        currentProps = self.currentQNet.properties()
        props = zip(targetProps, currentProps)
        return [targetVar.assign(currVar) for targetVar, currVar in props]


    def selectAction(self, currentState):
        action = np.zeros(len(self.actions))
        if random.random() < self.epsilon:
            actionInd = random.randrange(0, len(self.actions))
        else:
            qOut = self.currentQNet.QValue.eval(feed_dict = { self.currentQNet.stateInput: [currentState] } )
            # TODO: choose the action based on qOut
        action[actionInd] = 1.0

        return action

    def storeExperience(self, state, action, reward, newState, terminalState):
        if len(self.replayMemory) > MAX_REPLAY_MEMORY:
            self.replayMemory.popleft()
        ##TODO: store experience in the replay buffer

    def sampleExperiences(self):
        if len(self.replayMemory) < BATCH_SIZE:
            return list(self.replayMemory)
        # TODO: randomly sample a buffer size sample from the replay buffer.




