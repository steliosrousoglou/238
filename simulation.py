from matplotlib import pyplot as plt
import numpy as np
import sys

# Probability that ball is returned
## key: (pos, spin) of ball
## value: probability map based on position ([TL, TR, BL, BR])
prob_ret = {('TL', 'TS'): {'TL': 0.7, 'TR': 0.1, 'BL': 0.7, 'BR': 0.3}, ('TL', 'SL'): {'TL': 0.6, 'TR': 0.7, 'BL': 0.6, 'BR': 0.4},\
            ('TR', 'TS'): {'TL': 0.1, 'TR': 0.7, 'BL': 0.3, 'BR': 0.7}, ('TR', 'SL'): {'TL': 0.7, 'TR': 0.6, 'BL': 0.4, 'BR': 0.6},\
            ('BL', 'TS'): {'TL': 0.4, 'TR': 0.2, 'BL': 0.7, 'BR': 0.4}, ('BL', 'SL'): {'TL': 0.7, 'TR': 0.5, 'BL': 0.6, 'BR': 0.7}, \
            ('BR', 'TS'): {'TL': 0.2, 'TR': 0.4, 'BL': 0.4, 'BR': 0.7}, ('BR', 'SL'): {'TL': 0.5, 'TR': 0.7, 'BL': 0.7, 'BR': 0.6}}

# Stochastic outcomes of ball position
## key: (pos) of player hitting the ball
## value: probability of ball going to [TL, TR, BL, BR]
prob_outcomes = {'TL': {'TL': 0.28, 'TR': 0.12, 'BL': 0.36, 'BR': 0.24}, \
                 'TR': {'TL': 0.12, 'TR': 0.28, 'BL': 0.24, 'BR': 0.36}, \
                 'BL': {'TL': 0.21, 'TR': 0.09, 'BL': 0.49, 'BR': 0.21}, \
                 'BR': {'TL': 0.09, 'TR': 0.21, 'BL': 0.21, 'BR': 0.49}}

# Stochastic outcome of serve
## key: 1 or 2 (first or second serve)
prob_serve = {1: 0.4, 2: 0.7}

def update_score(p, points_p0, points_p1):
    if p == 0:
        return points_p0 + 1, points_p1
    return points_p0, points_p1 + 1

def update_player(p):
    return abs(p - 1)

# Return stochastic outcome of ball position
def pick_outcome(pos):
    # given player position, return stochastic outcome
    probs = prob_outcomes[pos]
    n = np.random.uniform()
    if n < probs['TL']:
        return 'TL'
    if n < probs['TL'] + probs['TR']:
        return 'TR'
    if n < probs['TL'] + probs['TR'] + probs['BL']:
        return 'BL'
    return 'BR'

# Return stochastic outcome of spin
def pick_spin(pos_p, pos_ball):
    if pos_p == pos_ball:
        if np.random.uniform < 0.8:
            return 'TS'
        return 'SL'
    else:
        if np.random.uniform < 0.8:
            return 'SL'
        return 'TS'

def smooth_skill(p):
    return np.power(1.015, p)

def smooth_speed(p):
    return np.power(1.04, p)

def choose_action_default(current_p, pos, pos_ball, ball_spin, params):
    if pos[current_p][0] != pos_ball[0] and pos[current_p][1] != pos_ball[1]:
        # move diagonally top/bottom, left/right - 60%
        if np.random.uniform() < 0.5 * smooth_speed(params[current_p + 2]):
            return pos_ball
        elif np.random.uniform() < 0.6 * smooth_speed(params[current_p + 2]):
            return pos_ball[0] + pos[current_p][1]
        elif np.random.uniform() < 0.4 * smooth_speed(params[current_p + 2]):
            return pos[current_p][0] + pos_ball[1]
        # else stay in position
    elif pos[current_p][0] == pos_ball[0]:  # both top or bottom
        if np.random.uniform() < 0.6 * smooth_speed(params[current_p + 2]):
            return pos_ball
    elif pos[current_p][1] == pos_ball[1]:  # both left or right
        if np.random.uniform() < 0.65 * smooth_speed(params[current_p + 2]):
            return pos_ball
    return pos[current_p]

# Play point and return updated score - current_p is RETURN player
def play_point(current_p, points_p0, points_p1, spin_ball, params, choose_action_0, choose_action_1):
    points = [points_p0, points_p1]
    # find position
    if sum(points) % 2 == 0:
        pos = ['BR', 'BR']
        pos_ball = 'TR'
    else:
        pos = ['BL', 'BL']
        pos_ball = 'TL'

    while True:
        # ball hits court at pos_ball, with spin_ball - current_p plays
        probs = prob_ret[(pos_ball, spin_ball)]

        # current_p moves towards ball (semi - stochastically)
        if current_p == 0:
            pos[current_p] = choose_action_0(current_p, pos, pos_ball, spin_ball, params)
        else: 
            pos[current_p] = choose_action_1(current_p, pos, pos_ball, spin_ball, params)

        # Does player return ball? If yes, where and how
        if np.random.uniform() < probs[pos[current_p]] * smooth_skill(params[current_p]):
            # update current player, ball position and spin
            spin_ball = pick_spin(pos[current_p], pos_ball)
            pos_ball = pick_outcome(pos[current_p])
            current_p = update_player(current_p)
        else:
            # point over!!
            points[update_player(current_p)] += 1
            return points

# Simulate match and return winner, set score
def simulate_match(params, strategy_0, strategy_1):
    sets_p0 = 0
    sets_p1 = 0

    serve = 1

    # Play match
    while sets_p0 + sets_p1 < 3:
        games_p0 = 0
        games_p1 = 0

        # Play Set
        while max(games_p0, games_p1) < 6 or games_p0 + games_p1 == 11:
            points_p0 = 0
            points_p1 = 0

            # next player serves
            serve = abs(serve - 1)
            # Play game
            while max(points_p0, points_p1) < 4:
                # Serve
                if (points_p0 + points_p1) % 2 == 0:
                    pos0 = pos1 = 'BR'
                else:
                    pos0 = pos1 = 'BL'

                if np.random.uniform() > prob_serve[1] * smooth_skill(params[serve + 4]):
                    # first serve out
                    if np.random.uniform() > prob_serve[2] * smooth_skill(params[serve + 4]):
                        # double fault, point to other player
                        points_p0, points_p1 = update_score(update_player(serve), points_p0, points_p1)
                        continue
                    else:
                        # play point, 2nd serve
                        points_p0, points_p1 = play_point(update_player(serve), points_p0, points_p1, 'SL', params, strategy_0, strategy_1)
                else:   
                    # play point, 1st serve
                    points_p0, points_p1 = play_point(update_player(serve), points_p0, points_p1, 'TS', params, strategy_0, strategy_1)

            # Declare game winner
            if points_p0 > points_p1:
                games_p0 += 1
            else:
                games_p1 += 1

        # Play tie-break
        if (games_p0 == games_p1 == 6):
            points_p0 = 0
            points_p1 = 0

            # Need 2-point lead to win set
            while max(points_p0, points_p1) < 7 or abs(points_p0 - points_p1) < 2:
                # Serve
                if (points_p0 + points_p1) % 2 == 0:
                    pos0 = pos1 = 'BR'
                else:
                    pos0 = pos1 = 'BL'
                    serve = abs(serve - 1)

                if np.random.uniform() > prob_serve[1] * smooth_skill(params[serve + 4]):
                    # first serve out
                    if np.random.uniform() > prob_serve[2] * smooth_skill(params[serve + 4]):
                        # double fault, point to other player
                        points_p0, points_p1 = update_score(update_player(serve), points_p0, points_p1)
                        continue
                    else:
                        # play point, 2nd serve
                        points_p0, points_p1 = play_point(update_player(serve), points_p0, points_p1, 'SL', params, strategy_0, strategy_1)
                else:
                    # play point, 1st serve
                    points_p0, points_p1 = play_point(update_player(serve), points_p0, points_p1, 'TS', params, strategy_0, strategy_1)

            # Declare tie-break (set) winner
            if points_p0 > points_p1:
                games_p0 += 1
            else:
                games_p1 += 1

        # Declare set winner
        if games_p0 > games_p1:
            sets_p0 += 1
        else:
            sets_p1 += 1

    # Declare winner
    if sets_p0 > sets_p1:
        return 0, sets_p0, sets_p1
    else:
        return 1, sets_p0, sets_p1

def run_simulation(iters, params, strategy_0=choose_action_default, strategy_1=choose_action_default):
    wins = [0, 0]
    y_wins = []
    for i in range(1, iters + 1):
        winner, sets_p0, sets_p1 = simulate_match(params, strategy_0, strategy_1)
        wins[winner] += 1
        if i % 10 == 0:
            y_wins.append(wins[0])
    return wins, y_wins

def main():
    # parametrize play by a and b, how good players are (Range: 1 - 10)
    iters = 1000
    params = [9, 8, 7, 9, 7, 10] # [p0 skill, p1 skill, p0 speed, p1 speed, p0 serve, p1 serve]
    wins = run_simulation(iters, params)
    print(wins)

if __name__ == "__main__":
    main()
