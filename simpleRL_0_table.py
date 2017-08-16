# from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np

# create OpenAI Frozen Lake gym
env = gym.make('FrozenLake-v0')

print("table lookup Q-learning\n")

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
print(np.shape(Q))
# Set learning parameters
lr = .85
y = .99
num_episodes = 100
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)

    if i % 500 == 0:
        print("Score over time: " +  str(sum(rList)/num_episodes))

print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
print("Final Q-Table Values")
print(Q)
