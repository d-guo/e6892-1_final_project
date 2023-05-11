import numpy as np

from bandits import EpsilonGreedy, LinUCB

p1 = ['player1', 25, 6, 6, []]
p2 = ['player2', 25, 7, 8, []]

user_contexts = [
    [0, 25, 0, 6, []],
    [1, 30, 0, 0, []],
    [2, 50, 0, 6, []]
]

bandit_alg = LinUCB(user_contexts, 4, 50)

print('type 0 to connect, 1 to decline')
i = 0
while True:
    if i < 6:
        player_context = p1
    else:
        player_context = p2

    print('-------------------------')
    print(f'playing as {player_context[0]}')

    print('-------------------------')
    # recommend
    arm_chosen = bandit_alg.choose_arm(player_context)

    print('-------------------------')
    # decide
    print(f'do you want to connect {arm_chosen}')
    reward = int(input())

    # update
    bandit_alg.update(arm_chosen, reward, player_context)
    i += 1