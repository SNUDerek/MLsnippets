# from https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149

import tensorflow as tf
import numpy as np

#List out our bandits:
bandits = [0.13,0.04,-0.02,-0.12]
num_bandits = len(bandits)

# function for bandit (slot machine)
# takes bandit #, returns reward
def pullBandit(bandit):
    #Get a random number. (sampled from norm dist)
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        #return a negative reward.
        return -1

tf.reset_default_graph()

weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights,0)

# data placeholders
reward_hold = tf.placeholder(shape=[1],dtype=tf.float32)
action_hold = tf.placeholder(shape=[1],dtype=tf.int32)
reponsibility = tf.slice(weights, action_hold,[1])
# loss = log(policy)*advantage
loss = -(tf.log(reponsibility)*reward_hold)

# network stuff
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
update = optimizer.minimize(loss)

maxiters = 10000

reward_sum = np.zeros(num_bandits)
rand_prob = 0.25 # random action prob

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)
    i = 0

    while i < maxiters:

        # choose random action sometimes else do selected action
        if np.random.rand(1) < rand_prob:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)

        reward = pullBandit(bandits[action])

        # network update
        _,resp,ww = sess.run([update,reponsibility,weights], feed_dict={reward_hold:[reward],action_hold:[action]})

        #Update our running tally of scores.
        reward_sum[action] += reward
        if i % 100 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits: " + str(reward_sum))
        i+=1
print("agent choosing bandit " + str(np.argmax(ww)+1) + " as best")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    
    print("...and it was right!")
else:
    print("...and it was wrong!")
