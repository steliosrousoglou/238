from collections import Counter
import operator
from simulation import prob_ret, prob_outcomes, \
    prob_serve, smooth_speed, smooth_skill, pick_spin, \
    pick_outcome, update_player, run_simulation

from matplotlib import pyplot as plt
from random import choice
import numpy as np
from sys import argv, exit

#max_Qs = Counter()
Q0 = Counter()
Q1 = Counter()
Q0_map = {}
Q1_map = {}

actions = ['BL', 'BR', 'TL', 'TR']

# State definition: (pos, pos_ball, spin_ball)
# Actions available: [TL, TR, BL, BR] - stochastic outcomes

# Execute action because outcome is stochastic
def execute_action(current_p, pos, params, a):
    if pos[0] != a[0] and pos[1] != a[1]:
        # move diagonally top/bottom, left/right - 60%
        if np.random.uniform() < 0.5 * smooth_speed(params[current_p + 2]):
            return a
        elif np.random.uniform() < 0.6 * smooth_speed(params[current_p + 2]):
            return a[0] + pos[1]
        elif np.random.uniform() < 0.4 * smooth_speed(params[current_p + 2]):
            return pos[0] + a[1]
        # else stay in position
    elif pos[0] == a[0]:  # both top or bottom
        if np.random.uniform() < 0.6 * smooth_speed(params[current_p + 2]):
            return a
    elif pos[1] == a[1]:  # both left or right
        if np.random.uniform() < 0.65 * smooth_speed(params[current_p + 2]):
            return a
    return pos

def update_Q(winner, p_hist, alpha):
    global Q0, Q1
    gamma = 0.9

    #for i, s in enumerate(p_hist[0]):
        # discount * reward * won or lost
    if len(p_hist[0]) > 0:
        Q0[p_hist[0][0]] += alpha * (3 * np.power(-1, winner) ) #* np.power(gamma, i)

    #for i, s in enumerate(p_hist[1]):
        # discount * reward * won or lost
    if len(p_hist[1]) > 0:
        Q1[p_hist[1][0]] += alpha * (3 * np.power(-1, 1 - winner) )# * np.power(gamma, i)

# Q0_map, Q1_map
# e.g. {(pos, ball_pos, ball_spin): {'TL':0.452, 'TR':0.13, 'BL':0.76, 'BR': 0.12}}
def tranform_Qs():
    global Q0, Q1, Q0_map, Q1_map

    for s in Q0:
        pos, ball, spin, a = s
        sp = (pos, ball, spin)
        if sp not in Q0_map:
            Q0_map[sp] = {}

        Q0_map[sp][a] = Q0[s]

    for s in Q1:
        pos, ball, spin, a = s
        sp = (pos, ball, spin)
        if sp not in Q1_map:
            Q1_map[sp] = {}

        Q1_map[sp][a] = Q1[s]

    return Q0_map, Q1_map

# Pla- y point update Q values along the way 
def play_point(params, alpha):
    p_hist = {0: [], 1:[]}
    # find position
    if np.random.uniform() < 0.5:
        pos = ['BR', 'BR']
        pos_ball = 'TR'
        current_p = 0
    else:
        pos = ['BL', 'BL']
        pos_ball = 'TL'
        current_p = 1
    if np.random.uniform() < 0.5:
        spin_ball = 'TS'
    else:
        spin_ball = 'SL'

    while True:
        # ball hits court at pos_ball, with spin_ball - current_p plays
        probs = prob_ret[(pos_ball, spin_ball)]

        # current_p moves towards ball (semi - stochastically)
        a = choice(actions)#explore_action(current_p, pos[current_p], pos_ball, spin_ball, params)
        pos_prev = pos[current_p]
        pos[current_p] = execute_action(current_p, pos[current_p], params, a)
        p_hist[current_p].insert(0, (pos_prev, pos_ball, spin_ball, a))

        # Does player return ball? If yes, where and how
        if np.random.uniform() < probs[pos[current_p]] * smooth_skill(params[current_p]):
            # update current player, ball position and spin
            spin_ball = pick_spin(pos[current_p], pos_ball)
            pos_ball = pick_outcome(pos[current_p])
            current_p = update_player(current_p)
        else:
            # point over!!
            winner = update_player(current_p)
            update_Q(winner, p_hist, alpha)
            return

# Q-learning with eligibility traces (lambda ~ 0.8)
def Q_learning(iters, params):
    global Q0, Q1
    alpha = 0.1

    for i in range(iters):
        play_point(params, alpha)

def choose_Q_action(current_p, pos, pos_ball, ball_spin, params):
    global Q0_map, Q1_map
    max_a = None
    max_a_val = - float('Inf')

    if current_p == 0:
        s = (pos[0], pos_ball, ball_spin)
        return max(Q0_map[s].iteritems(), key=operator.itemgetter(1))[0]
    else:
        s = (pos[1], pos_ball, ball_spin)
        return max(Q1_map[s].iteritems(), key=operator.itemgetter(1))[0]

def test_action(current_p, pos, pos_ball, ball_spin, params):
    return pos_ball

def differences(params):
    global Q0_map, Q1_map

    for s in Q0_map:
        if s not in Q1_map:
            continue
        a0 = choose_Q_action(0, [s[0], None], s[1], s[2], params)
        a1 = choose_Q_action(1, [None, s[0]], s[1], s[2], params)

        if a0 != a1:
            print(s)
            print(a0)
            print(a1)
            print('------------------------')

def main():
    global Q0_map, Q1_map
    if len(argv) != 2:
        raise Exception("usage: python project2.py <iterations>")

    its = int(argv[1])

    params = [8,8,8,4,8,8]

    ### Q-learning-opponent ###

    Q_learning(its, params)
    
    Q0_map, Q1_map = tranform_Qs()

    # Run some tests
    iters = 1000

    wins_naive, naive_y = run_simulation(iters, params)
    wins_Q, Q_y = run_simulation(iters, params, choose_Q_action, choose_Q_action)
    
    print(wins_naive)
    print(wins_Q)
    print(Q_y)

    X = np.linspace(10, 1000, 100)

    g = plt.scatter(X, np.divide(np.array(naive_y), X), alpha=0.7, marker='.')
    d = plt.scatter(X, np.divide(np.array(Q_y), X), alpha=0.7, marker='.')
    plt.xlabel("number of simulations")
    plt.ylabel("fraction of wins")
    plt.title('Fraction of wins for Null and RL models - ' + str(its) + ' iterations')
    plt.text(650, 0.02, 'params: ' + str(params))
    plt.legend((g, d), ('NULL', 'RL'), fontsize=15)
    plt.xlim(0, 1000)
    plt.ylim(0, 1)
    plt.show()

    # Print differences in strategies learned
    differences(params)

    '''
    ## Q models only ###
        # Run some tests
    iters = 1000

    wins_naive, naive_y = run_simulation(iters, params)
    
    print(wins_naive)

    X = np.linspace(10, 1000, 100)

    g = plt.scatter(X, np.divide(np.array(naive_y), X), alpha=0.7, marker='.')
    #d = plt.scatter(X, np.divide(np.subtract(X, np.array(wins_naive[0])), X), alpha=0.7, marker='.')
    plt.xlabel("number of simulations")
    plt.ylabel("fraction of wins")
    plt.title('Fraction of wins for Player 1 (Null models)')
    plt.text(650, 0.02, 'params: ' + str(params))
    #plt.legend((g, d), ('NULL', 'RL'), fontsize=15)
    plt.xlim(0, 1000)
    plt.ylim(0, 1)
    plt.show()
    '''
if __name__ == '__main__':
    main()