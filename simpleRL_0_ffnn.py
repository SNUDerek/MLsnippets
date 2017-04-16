# from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np
import tensorflow as tf

# create OpenAI Frozen Lake gym
env = gym.make('FrozenLake-v0')

print("neural network Q-learning\n")

tf.reset_default_graph()

# feed-forward network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0.001,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

# sum of squares loss between target (Bellman eqn) and prediction Q vals.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

##########
# training
##########

init = tf.initialize_all_variables()

# Set learning parameters
y = .75 # discount factor, near 1 so favoring long-term
e = 0.5 # random action rate
num_episodes = 5000

#create lists to contain total rewards and steps per episode
jList = []
rList = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False # 'done' boolean (terminate sim)
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model
                if e > 0.1:
                    e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
        if i % 500 == 0:
            print("Score over time: " +  str(sum(rList)/num_episodes))
        
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
