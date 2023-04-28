import numpy as np
import pulp
import math
import argparse

def binary_to_string(num, len_action):
    binary_string = "{0:b}".format(num)
    # print(binary_string)
    binary_string = "0"*(len_action-len(binary_string))+binary_string
    # print(binary_string)
    string = ""
    for char in binary_string:
        if char == '0':
            string += 'R'
        else:
            string += 'B'
    return string
    # print(string)

def generate_states(WINDOW_LEN):
    states0 = ['0']
    for w in range(1, WINDOW_LEN+1):
        for i in range(2**w):
            states0.append('0'+binary_to_string(i, w))
    states1 = ['1']
    for w in range(1, WINDOW_LEN+1):
        for i in range(2**w):
            states1.append('1'+binary_to_string(i, w))
    return states0, states1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", default = None)
    parser.add_argument("--window_len", default = None)
    parser.add_argument("--seed", default = 0)

    args = parser.parse_args()
    np.random.seed(int(args.seed))

    K = args.K
    WINDOW_LEN = args.window_len
    if K == None:
        K = 0.1
    else:
        K = float(K)
    if WINDOW_LEN == None:
        WINDOW_LEN = 1
    else:
        WINDOW_LEN = int(WINDOW_LEN)
    numStates = 2**(WINDOW_LEN + 2) - 2       
    numActions = 4
    gamma = 0.5
    p0 = np.random.rand()
    p1 = np.random.rand()
    p2 = np.random.rand()
    p3 = np.random.rand()
    Tr= np.array([[p0, 1-p0],
                [p1, 1-p1]])

    Tb = np.array([[p2, 1-p2],
                [p3, 1-p3]])

    Cr = np.random.rand(2) #np.array([0,1])
    Cb = np.random.rand(2) #np.array([1,0])
    Crs = Cr+gamma*K
    Cbs = Cb+gamma*K

    T = {}
    T['R'] = Tr
    T['B'] = Tb
    T['RS'] = Tr
    T['BS'] = Tb

    C = {}
    C['R'] = Cr
    C['B'] = Cb
    C['RS'] = Crs
    C['BS'] = Cbs

    # actions = R, B, RB, RR, BR, BB
    print("numStates", numStates)
    print("numActions", numActions)
    print("end -1")

    states0, states1 = generate_states(WINDOW_LEN)
    actions = ['R', 'B', 'RS', 'BS']
    mdp = {state: {action: {} for action in actions} for state in states0+states1}
    belief = {"0":np.array([1, 0]), "1":np.array([0, 1])}

    for state in [0,1]:
        for action in ['RS', "BS"]:
            for nextState in [0, 1]:
                mdp[str(state)][action][str(nextState)] = {}
                prob = T[action][state][nextState]
                cost = C[action][state]
                mdp[str(state)][action][str(nextState)]['prob'] = prob
                mdp[str(state)][action][str(nextState)]['cost'] = cost

    for state in states0[1:]+states1[1:]:
        belief[state] = belief[str(state[:-1])]@T[state[-1]]
        for action in ["RS", "BS"]:
            for nextState in [0, 1]:
                mdp[state][action][str(nextState)] = {}
                prob = (belief[state]@T[action])[nextState]
                cost = belief[state]@C[action]
                mdp[state][action][str(nextState)]['prob'] = prob
                mdp[state][action][str(nextState)]['cost'] = cost

    for state in states0+states1:
        if(len(state) <= WINDOW_LEN):
            for action in ["R", "B"]:
                nextState = state + action
                prob = 1
                cost = belief[state]@C[action]
                mdp[state][action][nextState] = {}
                mdp[state][action][nextState]['prob'] = prob
                mdp[state][action][nextState]['cost'] = cost


    for state in states0+states1:
        for action in actions:
            for nextState in mdp[state][action].keys():
                cost = mdp[state][action][nextState]['cost']
                prob = mdp[state][action][nextState]['prob']
                print("transition ", state, " ", action, " ", nextState, " ", cost, " ", prob)

    print("gamma ", gamma)
    # print belief dictionary into a text file
    with open("belief.txt", "w") as f:
        for key in belief.keys():
            f.write("%s %s  " % (key, belief[key][0]))
            f.write("\n")
