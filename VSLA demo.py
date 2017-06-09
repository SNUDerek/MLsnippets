# example of a variable-state learning automaton
# simple state-output model
# see Learning Automata: An Introduction

# CITATIONS:
# https://theses.lib.vt.edu/theses/available/etd-5414132139711101/unrestricted/ch3.pdf
# CDF code: http://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable
# VSLA outline: https://www.researchgate.net/figure/225274789_fig10_Figure-2-Pseudo-code-of-variable-structure-learning-automaton

import random
# state-output (G = H) variable-state LA

i = 1       # iterator
s = 'null'     # user input
items = []  # items list
probs = []  # probs list
beta = 0    # learning parameter
param = 0.25

items = ['hunting meat', 'gathering fruit', 'catching fish', 'eating dirt']

# initialize p to [1/r, 1/r, 1/r,...,1/r]
r = len(items)
for item in items:
    probs.append(1/r)

# while not done:
while(s != 'x'):

    iter = 1

    # print probs
    print("")
    for i in range(0,len(probs)):
        print('Pr {:20}:{}'.format(items[i], probs[i]))
    print('{:23}:{}'.format('Total:', sum(probs)))

    # select action i based on prob vector p
    # choose next state based on state probabilities (CDF):
    # (this is bc state-output and so only current state matters)
    randy = random.random() # random float from 0-1
    index = 0
    # step through probs, stop at index that contains r
    while(randy >= 0 and index < len(probs)):
        randy -= probs[index]
        index += 1

    state = (index - 1)

    # evaluate action and return reinforcement signal beta
    print("")
    print(iter,": today I am", items[state], "for food. Did I eat well? (y/n/x)")
    s = input()

    # just for ease of visualization
    if s is 'x':
        break
    elif s is 'y':
        beta = 0
    elif s is 'n':
        beta = 1
    else:
        print('invalid input, setting to N')
        beta = 1

    # update probability vector using learning alg
    # Linear Reward-Penalty scheme (L_R-P)
    # see Learning Automata: An Introduction
    for j in range(0,len(probs)):

        # beta = 0 (not punished)
        if beta == 0:
            # update probs
            if j == state:
                probs[j] = probs[j] + param * (1 - probs[j])
            else:
                probs[j] = (1 - param) * probs[j]

        # beta = 1 (punished!!!!)
        elif beta == 1:
            # update probs
            if j == state:
                probs[j] = (1-param) * probs[j]
            else:
                probs[j] = (param / (r - 1))+((1-param)*probs[j])

        else:
            print('something broke... :(')
            probs = probs

    iter += 1


