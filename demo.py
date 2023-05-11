import numpy as np

from bandits import EpsilonGreedy, LinUCB

p1 = ['player1', 0, 6, 6, []]
p2 = ['player2', 50, 6, 6, []]

user_contexts = [
    [0, 5, 0, 0, []],
    [1, 55, 0, 0, []]
]

bandit_alg = LinUCB(user_contexts, 4, 500)

print('type 1 to connect, 0 to decline')
print('-------------------------')
print('players')
print('p1 is', p1)
print('p2 is', p2)
print('-------------------------')
print('rec options (arms)')
for i in range(len(user_contexts)):
    print(user_contexts[i])
i = 0
while True:
    if i < 4:
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