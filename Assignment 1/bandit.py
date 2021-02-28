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
        print("Please Print a Vaild Algorithm\n")
        exit()

## Modelling the Bernoulli rewards        
def bernoulli_reward(p):
    if (np.random.rand() <= p):
        return 1.
    else:
        return 0.

def find_max_q(p, u, t):
    q = 0
    RHS = (np.log(t) + 3 * np.log(np.log(t))) / u
    i = 0.1
    while (i < 1):
        LHS = KL_Divergence(p, i)
        if (LHS <= RHS):
            q = i
        i += 0.05
    return q
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
        # makes a decision for which arm to pull
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
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret


def ucb(mean_rewards, horizon):
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
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
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
            ucb_list[arm] = find_max_q(calculated_means[arm], num_pulls[arm], i)
        # picking the arm with maximum KL-UCB
        arm_idx = np.argmax(ucb_list)
        reward = bernoulli_reward(mean_rewards[arm_idx])
        expected_reward += reward
        calculated_means[arm_idx] = (num_pulls[arm_idx] * calculated_means[arm_idx] + reward) / (num_pulls[arm_idx] + 1)
        num_pulls[arm_idx] += 1
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
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
           # alpha = arm_success + 1, beta = arm_failure + 1
           betas[arm] = np.random.beta(arm_success + 1, arm_failure + 1)
        # picks an arm with the maximum Beta value
        arm_idx = np.argmax(betas)
        reward = bernoulli_reward(mean_rewards[arm_idx])
        if (reward == 0):
            # failure occurs with reward 0
            failures[arm_idx] += 1
        else:
            # success occurs with reward 1
            successes[arm_idx] += 1
        expected_reward += reward
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret

def run_as_function(banditFile, algorithm, seed, epsilon, horizon):
    np.random.seed(seed)
    mean_rewards = []
    with open(banditFile, 'r') as f:
        for line in f:
            mean_rewards.append(float(line))
    regret = run_algorithm(mean_rewards, algorithm, epsilon, horizon)
    output_string = [banditFile, algorithm, str(seed), str(epsilon), str(horizon), str(regret)]
    output_string = ", ".join(output_string)
    print(output_string)

def thompson_sampling_with_hint(mean_rewards, horizon):
    # pulls all arms in a round-robin manner
    num_arms = len(mean_rewards)
    
    hint_ls=np.sort(mean_rewards)

    # expected_reward = 0
    # for i in range(len(mean_rewards)):
    #     arm_idx = i 
    #     reward = bernoulli_reward(mean_rewards[arm_idx])
    #     expected_reward += reward
    ideal_reward = max(mean_rewards) * horizon
    num_arms = len(mean_rewards)
    expected_reward = 0
    successes = [0] * num_arms
    failures = [0] * num_arms
    betas = [0] * num_arms
    means=[0]*num_arms
    for i in range(horizon):


    	
        for arm in range(num_arms):
           arm_success = successes[arm]
           arm_failure = failures[arm]
           # picks a number from the Beta distribution with 
           # alpha = arm_success + 1, beta = arm_failure + 1
           betas[arm] = np.random.beta(arm_success + 1, arm_failure + 1)
           means[arm]=  (arm_success + 1)/(arm_success+arm_failure+2)
        # picks an arm with the maximum Beta value
        t=0
        safe_set=[0]* num_arms
        for mean_val in betas:
        	if (mean_val>=hint_ls[0] ):
        		safe_set[t]=betas[t]
        	t=+1 

        if (np.any(safe_set)):
        	arm_idx = np.argmax(means)
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
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret

def run_as_function(banditFile, algorithm, seed, epsilon, horizon):
    np.random.seed(seed)
    mean_rewards = []
    with open(banditFile, 'r') as f:
        for line in f:
            mean_rewards.append(float(line))
    regret = b_algorithm(mean_rewards, algorithm, epsilon, horizon)
    output_string = [banditFile, algorithm, str(seed), str(epsilon), str(horizon), str(regret)]
    output_string = ", ".join(output_string)
    print(output_string)
    #return regret

instances = ["instances/i-1.txt", "instances/i-2.txt", "instances/i-3.txt"]
algorithms = ["thompson-sampling",'thompson-sampling-with-hint']
horizons = [100,400,1600,6400,25600,102400]
#epsilons=[0.0002,0.002,0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for seed in range(50):
	for instance in instances:
		for horizon in horizons:
			for algorithm in algorithms:
				run_as_function(instance, algorithm, seed, 0, horizon)

	
	# for instance in instances:
	# 	print("instance "+str(instance))
	# 	i=0
	# 	for seed in range(50):

	# 		i=(i+run_as_function(instance, "epsilon-greedy", seed, ep, 102400))/50
	# 	print(i)



# if __name__ == "__main__":
#     banditFile = str(sys.argv[1])
#     algorithm =str( sys.argv[2])
#     seed = int(sys.argv[3])
#     epsilon = float(sys.argv[4])
#     horizon = int(sys.argv[5])
#     np.random.seed(int(seed))
#     mean_rewards = []
#     with open(banditFile, 'r') as f:
#         for line in f:
#             mean_rewards.append(float(line))
#     regret = b_algorithm(mean_rewards, algorithm, epsilon, horizon)
#     output_string = [banditFile, algorithm, str(seed), str(epsilon), str(horizon), str(regret)]
#     output_string = ", ".join(output_string)
#     print(output_string)