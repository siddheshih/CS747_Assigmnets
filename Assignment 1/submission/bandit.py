import numpy as np
import sys



## Which Algorithm to run to generate data
def b_algorithm(mean_rewards, algorithm, epsilon, horizon):
    if (algorithm == "thompson-sampling-with-hint"):
        return thompson_sampling_with_hint(mean_rewards, horizon)
    elif (algorithm == "epsilon-greedy"):
        return epsilon_greedy(mean_rewards, epsilon, horizon)
    elif (algorithm == "ucb"):
        return ucb(mean_rewards, horizon)
    elif (algorithm == "kl-ucb"):
        return kl_ucb(mean_rewards, horizon)
    elif (algorithm == "thompson-sampling"):
        return thompson_sampling(mean_rewards, horizon)
    else:
        print("Please enter a algorithm with valid format\n")
        exit()

## Modelling the Bernoulli rewards        
def bernoulli_reward(p):
    if (np.random.rand() <= p):
        return 1.
    else:
        return 0.

##This function is used to calculate the Q used in KL ucb algorithm
##This uses the well known iteration to get the Qmax
def Q_max(p, u, t):
    Q = 0
    RHS = (np.log(t) + 3 * np.log(np.log(t))) / u
    i = 0.1
    while (i < 1):
        LHS = KL_Divergence(p, i)
        if (LHS <= RHS):
            Q = i
        i += 0.05
    return Q
## KL_Divergence
def KL_Divergence(x, y):
    if (x == 0):
        return np.log(1 / (1 - y))
    elif (x == 1):
        return np.log(1 / y)
    else:
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def epsilon_greedy(mean_rewards, epsilon, horizon):
    num_arms = len(mean_rewards)
    calculated_means = [0] * num_arms
    num_pulls = [0] * num_arms
    expected_reward = 0
    for i in range(horizon):
        arm_idx = 0
        # The deciiosn variable which decises wheter to explore or explit
        decision = np.random.choice([1, 2], p = [epsilon, 1 - epsilon])
        if (decision == 1):
            # randomly pick an arm to pull
            arm_idx = mean_rewards.index(np.random.choice(mean_rewards))
        else:
            # pick arm with max mean
            arm_idx = np.argmax(calculated_means)
        reward = bernoulli_reward(mean_rewards[arm_idx])
        expected_reward += reward
        # calculates the new mean for the pulled arm
        calculated_means[arm_idx] = (num_pulls[arm_idx] * calculated_means[arm_idx] + reward) / (num_pulls[arm_idx] + 1)
        num_pulls[arm_idx] += 1
    optimal_reward = max(mean_rewards) * horizon
    regret = round(optimal_reward - expected_reward, 3)
    return regret


def ucb(mean_rewards, horizon):
    num_arms = len(mean_rewards)
    calculated_means = [0] * num_arms
    num_pulls = [0] * num_arms
    ucb_list = [0] * num_arms
    expected_reward = 0
    # sampling each arm once- Basically doing the round robin sampling
    for arm in range(num_arms):
        reward = bernoulli_reward(mean_rewards[arm])
        num_pulls[arm] += 1
        calculated_means[arm] = reward
    for i in range(num_arms, horizon):
        # calculating the UCBs for each arm
        for arm in range(num_arms):
            p = calculated_means[arm]
            u = num_pulls[arm]
            ucb_list[arm] = p + np.sqrt(2 * np.log(i) / u)
        # picking the arm with maximum UCB
        arm_idx = np.argmax(ucb_list)
        reward = bernoulli_reward(mean_rewards[arm_idx])
        expected_reward += reward
        calculated_means[arm_idx] = (num_pulls[arm_idx] * calculated_means[arm_idx] + reward) / (num_pulls[arm_idx] + 1)
        num_pulls[arm_idx] += 1
    optimal_reward = max(mean_rewards) * horizon
    regret = round(optimal_reward - expected_reward, 3)
    return regret

def kl_ucb(mean_rewards, horizon):
    num_arms = len(mean_rewards)
    calculated_means = [0] * num_arms
    num_pulls = [0] * num_arms
    ucb_list = [0] * num_arms
    expected_reward = 0
    # sampling each arm once
    for arm in range(num_arms):
        reward = bernoulli_reward(mean_rewards[arm])
        num_pulls[arm] += 1
        calculated_means[arm] = reward
    for i in range(num_arms, horizon):
        # calculating the KL-UCBs of the arms
        for arm in range(num_arms):
            p = calculated_means[arm]
            u = num_pulls[arm]
            ucb_list[arm] = Q_max(calculated_means[arm], num_pulls[arm], i)
        # picking the arm with maximum KL-UCB
        arm_idx = np.argmax(ucb_list)
        reward = bernoulli_reward(mean_rewards[arm_idx])
        expected_reward += reward
        calculated_means[arm_idx] = (num_pulls[arm_idx] * calculated_means[arm_idx] + reward) / (num_pulls[arm_idx] + 1)
        num_pulls[arm_idx] += 1
    optimal_reward = max(mean_rewards) * horizon
    regret = round(optimal_reward - expected_reward, 3)
    return regret


def thompson_sampling(mean_rewards, horizon):
    num_arms = len(mean_rewards)
    expected_reward = 0
    successes = [0] * num_arms
    failures = [0] * num_arms
    betas = [0] * num_arms
    for i in range(horizon):
        for arm in range(num_arms):
           arm_success = successes[arm]
           arm_failure = failures[arm]
           # picks a number from the Beta distribution with 
           
           betas[arm] = np.random.beta(arm_success + 1, arm_failure + 1)
        # picks an arm with the maximum Beta value
        arm_idx = np.argmax(betas)
        reward = bernoulli_reward(mean_rewards[arm_idx])
        if (reward == 0):
            # failure if the reward is zero
            failures[arm_idx] += 1
        else:
            # success if reward is 1
            successes[arm_idx] += 1
        expected_reward += reward
    optimal_reward = max(mean_rewards) * horizon
    regret = round(optimal_reward - expected_reward, 3)
    return regret

# def run_as_function(file_name, algorithm, seed, epsilon, horizon):
#     np.random.seed(seed)
#     mean_rewards = []
#     with open(file_name, 'r') as f:
#         for line in f:
#             mean_rewards.append(float(line))
#     regret = run_algorithm(mean_rewards, algorithm, epsilon, horizon)
#     output_string = [file_name, algorithm, str(seed), str(epsilon), str(horizon), str(regret)]
#     output_string = ", ".join(output_string)
#     print(output_string)

def thompson_sampling_with_hint(mean_rewards, horizon):
    # pulls each arm  once
    
    
    hint_ls=np.sort(mean_rewards)

    
    optimal_reward = max(mean_rewards) * horizon
    num_arms = len(mean_rewards)
    expected_reward = 0
    successes = [0] * num_arms
    failures = [0] * num_arms
    betas = [0] * num_arms
    means=[0]*num_arms
    var=[0]*num_arms


    for i in range(horizon):


        
        for arm in range(num_arms):
           arm_success = successes[arm]
           arm_failure = failures[arm]
           
           betas[arm] = np.random.beta(arm_success + 1, arm_failure + 1)
           means[arm]=  (arm_success + 1)/(arm_success+arm_failure+2)
           #var[arm]=((arm_success + 1)*(arm_failure + 1.))/((arm_success+arm_failure+2.)*(arm_success+arm_failure+2.)*(arm_success+arm_failure+3)) 
        # picks an arm with the maximum Beta value
        
        # safe_set=[0]* num_arms
        safe_set=[0]*num_arms
        for i in range(len(mean_rewards)):

            if (means[i]>=(hint_ls[-2]+((hint_ls[-1]-hint_ls[-2])/2))):
                safe_set[i]=means[i]
             

        if (np.any(safe_set)):
            arm_idx = np.argmax(safe_set)
        else:
            arm_idx = np.argmax(betas)
        reward = bernoulli_reward(mean_rewards[arm_idx])
        if (reward == 0):
            # failure occurs with reward 0
            failures[arm_idx] += 1
        else:
            # success occurs with reward 1
            successes[arm_idx] += 1
        expected_reward += reward
    optimal_reward = max(mean_rewards) * horizon
    regret = round(optimal_reward - expected_reward, 3)
    return regret

if __name__ == "__main__":
    file_name = str(sys.argv[2])
    algorithm =str( sys.argv[4])
    seed = int(sys.argv[6])
    epsilon = float(sys.argv[8])
    horizon = int(sys.argv[10])
    np.random.seed(int(seed))
    mean_rewards = []
    with open(file_name, 'r') as f:
        for line in f:
            mean_rewards.append(float(line))
    regret = b_algorithm(mean_rewards, algorithm, epsilon, horizon)
    output_string = [file_name, algorithm, str(seed), str(epsilon), str(horizon), str(regret)]
    output_string = ", ".join(output_string)
    print(output_string)