import numpy as np

def create_context(user_context, arm_context):
    age = 1 if np.abs(user_context[1] - arm_context[1]) <= 5 else -1
    occupation = np.abs(user_context[2] - arm_context[2]) == 0
    location = np.abs(user_context[3] - arm_context[3]) <= 1
    friends_already = arm_context[0] in user_context[4]

    return np.array([age, occupation, location, friends_already]).astype(int)


class EpsilonGreedy:

    def __init__(self, epsilon, num_arms):
        self.epsilon = epsilon
        self.num_arms = num_arms

        self.arms_reward = np.zeros(num_arms)
        self.arms_chosen = np.zeros(num_arms)

    def choose_arm(self, user_context):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            return np.argmax(self.arms_reward)
        
    def update(self, arm_chosen, reward, user_context):
        self.arms_chosen[arm_chosen] += 1
        self.arms_reward[arm_chosen] += (reward - self.arms_reward[arm_chosen]) / self.arms_chosen[arm_chosen]


class LinUCB:

    def __init__(self, arm_contexts, context_size, alpha):
        self.arm_contexts = arm_contexts
        self.context_size = context_size
        self.alpha = alpha
        self.num_arms = len(arm_contexts)

        self.A = [np.identity(context_size) for _ in range(self.num_arms)]
        self.b = [np.zeros(context_size) for _ in range(self.num_arms)]

    def choose_arm(self, user_context):
        max_ucb = float("-inf")
        best_arm = None
        for arm in range(self.num_arms):
            context = create_context(user_context, self.arm_contexts[arm])

            A_inv = np.linalg.inv(self.A[arm])
            theta_hat = np.dot(A_inv, self.b[arm])

            x = context.reshape((-1, 1))
            ucb = np.dot(theta_hat.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))
            print('arm:', arm, 'est reward:', np.dot(theta_hat.T, x))
            with open('log.txt', 'a') as f:
                f.write('arm: ' + str(arm) + str(theta_hat) + '\n')

            print(ucb)
            if ucb > max_ucb:
                max_ucb = ucb
                best_arm = arm
        
        return best_arm

    def update(self, arm_chosen, reward, user_context):
        context = create_context(user_context, self.arm_contexts[arm_chosen])
        x = context.reshape((-1, 1))
        self.A[arm_chosen] += np.dot(x, x.T)
        self.b[arm_chosen] += reward * x.flatten()
    