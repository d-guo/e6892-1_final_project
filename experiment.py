import numpy as np

from bandits import EpsilonGreedy, LinUCB

def create_context(user_context, arm_context):
    age = np.abs(user_context[1] - arm_context[1]) <= 5
    occupation = np.abs(user_context[2] - arm_context[2]) == 0
    location = np.abs(user_context[3] - arm_context[3]) <= 1
    friends_already = arm_context[0] in user_context[4]

    return np.array([age, occupation, location, friends_already]).astype(int)

user_contexts = [
    [i, np.random.randint(0, 9), np.random.randint(0, 9), np.random.randint(0, 9), []] for i in range(10)
]

linucb = LinUCB(user_contexts, 4, 5)
epsilongreedy = EpsilonGreedy(0.1, len(user_contexts))

rewards_linucb = []
rewards_epsilongreedy = []

for t in range(1000):
    user = np.random.randint(0, 10)
    user_context = user_contexts[user]

    arm = np.random.randint(0, 10)
    arm_context = user_contexts[arm]

    reward = np.sum(create_context(user_context, arm_context))
    reward = 1 if np.random.randint(1, 4) <= reward else 0

    arm1 = linucb.choose_arm(user_context)
    arm2 = epsilongreedy.choose_arm(user_context)

    if arm == arm1:
        rewards_linucb.append(reward)
        linucb.update(arm, reward, user_context)

    if arm == arm2:
        rewards_epsilongreedy.append(reward)
        epsilongreedy.update(arm, reward, user_context)


print(np.mean(rewards_linucb))
print(np.mean(rewards_epsilongreedy))