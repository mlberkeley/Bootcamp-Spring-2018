
from dqn import *
from cartpole import CartPole
import argparse

T = 1000000
UPDATE_TIME = 100

def evaluate(index):
    game = CartPole()
    actions = game.legal_actions
    dqn = DQN(actions)
    dqn.epsilon = 0
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("networks")
    if checkpoint:
        saver.restore(sess, checkpoint.all_model_checkpoint_paths[index])
        print "Loaded: %s" % checkpoint.all_model_checkpoint_paths[index]
    rewards = []
    for episode in range(200):
        state = game.newGame()
        totReward = 0
        for _ in range(400):
            if episode == 199:
                game.env.render()
            action = dqn.selectAction(state)
            actionNum = np.argmax(action)
            next_state, reward, game_over = game.next(actionNum)
            totReward += reward
            state = next_state
            if game_over:
                break
        rewards.append(totReward)
    print rewards
    print "Average %s, best %s" % (sum(rewards) / len(rewards), max(rewards))

def train():
    game = CartPole()
    actions = game.legal_actions
    dqn = DQN(actions)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    state = game.newGame()
    for episode in range(T):
        action = dqn.selectAction(state)
        actionNum = np.argmax(action)
        game.env.render()
        next_state, reward, game_over = game.next(actionNum)
        if game_over:
            dqn.storeExperience(state, action, 0, next_state, game_over)
            next_state = game.newGame()
        else:
            dqn.storeExperience(state, action, reward, next_state, game_over)

        ##TODO: sample a minibatch from the replay buffer
        state_batch = ???
        nextState_batch =???
        action_batch =???
        terminal_batch =???
        reward_batch =???

        y_batch = []
        Q_batch = sess.run(dqn.targetQNet.QValue, feed_dict = {dqn.targetQNet.stateInput: nextState_batch} )
        for i in range(len(minibatch)):
            terminal = terminal_batch[i]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                ## TODO: add the target to the list of targets for each element in the minibatch using Q update rule
                y_batch.append(???)
        currentQ_batch = sess.run(dqn.currentQNet.QValue,
                                  feed_dict = {dqn.currentQNet.stateInput: state_batch })

        sess.run(dqn.trainStep, feed_dict = {dqn.yInput: y_batch, dqn.actionInput: action_batch, dqn.currentQNet.stateInput: state_batch})
        state = next_state

        if episode % UPDATE_TIME == 0:
            sess.run(dqn.copyCurrentToTargetOperation())

        if episode % 25000 == 0:
            saver.save(sess, 'networks/' + 'dqn', global_step= episode)
        if dqn.epsilon > FINAL_EPSILON:
            ## TODO: decay epsilon which represents the probability of taking a random action
            dqn.epsilon -= 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    if args.test:
        evaluate(-1)
    else:
        train()
