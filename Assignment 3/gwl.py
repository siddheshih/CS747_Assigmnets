
import numpy as np
import matplotlib.pyplot as plt 


def get_nextstate_reward(rows, cols, wind, start, end_state, current_state, a, stochasticity_wind):
	reward = -1
	if stochasticity_wind == 0:
		wind_value = wind[int(current_state[1])]
	else: 
		wind_value = wind[int(current_state[1])] + np.random.randint(-1, 2)

	next_state = np.asarray([0, 0], dtype=int)

	if a == 0:
		next_state[0] = min(max(current_state[0] - 1 - wind_value, 0),  rows-1)
		next_state[1] = min(max(current_state[1], 0),  cols-1)

	elif a == 1:
		next_state[0] = min(max(current_state[0] + 1 - wind_value, 0),  rows-1)
		next_state[1] = min(max(current_state[1], 0),  cols-1)

	elif a == 2:
		next_state[0] = min(max(current_state[0] - wind_value, 0),  rows-1)
		next_state[1] = min(max(current_state[1] + 1, 0),  cols-1)
	
	elif a == 3:
		next_state[0] = min(max(current_state[0] - wind_value, 0),  rows-1)
		next_state[1] = min(max(current_state[1] - 1, 0),  cols-1)

	elif a == 4: 
		next_state[0] = min(max(current_state[0] - 1 - wind_value, 0),  rows-1)
		next_state[1] = min(max(current_state[1] + 1, 0),  cols-1)

	elif a == 5:
		next_state[0] = min(max(current_state[0] - 1 - wind_value, 0),  rows-1)
		next_state[1] = min(max(current_state[1] - 1, 0),  cols-1)

	elif a == 6:
		next_state[0] = min(max(current_state[0] + 1 - wind_value, 0),  rows-1)
		next_state[1] = min(max(current_state[1] + 1, 0),  cols-1)

	elif a == 7:
		next_state[0] = min(max(current_state[0] + 1 - wind_value, 0),  rows-1)
		next_state[1] = min(max(current_state[1] - 1, 0),  cols-1)

	if next_state[0] == end_state[0] and next_state[1] == end_state[1]:
		reward = 0

	return next_state, reward

def policy(current_state, epsilon, num_actions, transitions):
	#np.random.seed(seed)
	num = np.random.choice([1, 2], p = [epsilon, 1 - epsilon])
	ac = 0
	if (num == 1):
		ac = np.random.choice(range(num_actions))
	else:
		ac = np.argmax(transitions[current_state[0]][current_state[1]])
	return ac

def sarsa0(rows,cols, num_actions, episodes_tot, wind, start, end_state, alpha, epsilon, runs, gamma, stochasticity):
	transitions = np.zeros(( rows,  cols, num_actions), dtype=float)
	steps_per_episode = np.zeros(episodes_tot, dtype=float)

	for n in range(runs):
		seed = n
		np.random.seed(seed)
		for x in range(episodes_tot):
			current_state = start
			action = policy(current_state, epsilon, num_actions, transitions)
			time_steps = 0

			while(1):
				nextstate, reward = get_nextstate_reward( rows,  cols, wind, start, end_state, current_state, action, stochasticity)
				take_action = policy(nextstate, epsilon, num_actions, transitions)
				transitions[current_state[0]][current_state[1]][action] += alpha*(reward + gamma*transitions[nextstate[0]][nextstate[1]][take_action] - transitions[current_state[0]][current_state[1]][action])
				current_state = nextstate
				action = take_action
				time_steps += 1
				if current_state[0] == end_state[0] and current_state[1] == end_state[1]:
					break;
			steps_per_episode[x] += time_steps 

		transitions = np.zeros(( rows,  cols, num_actions), dtype=float)

	steps_per_episode = np.cumsum(steps_per_episode)
	return (steps_per_episode)/runs

def Q_learning(rows,cols, num_actions, episodes_tot, wind, start, end_state, alpha, epsilon, runs, gamma, stochasticity):
	transitions = np.zeros(( rows,  cols, num_actions), dtype=float)
	steps_per_episode = np.zeros(episodes_tot, dtype=float)

	for n in range(runs):
		seed = n
		np.random.seed(seed)
		for x in range(episodes_tot):
			current_state = start
			action = policy(current_state, epsilon, num_actions, transitions)
			time_steps = 0

			while(1):
				nextstate, reward = get_nextstate_reward( rows,  cols, wind, start, end_state, current_state, action, stochasticity)
				take_action = policy(nextstate, epsilon, num_actions, transitions)
				p=np.max(transitions[nextstate[0]][nextstate[1]])

				transitions[current_state[0]][current_state[1]][action] += alpha*(reward + gamma*p - transitions[current_state[0]][current_state[1]][action])
				current_state = nextstate
				action = take_action
				time_steps += 1
				if current_state[0] == end_state[0] and current_state[1] == end_state[1]:
					break;
			steps_per_episode[x] += time_steps 

		transitions = np.zeros(( rows,  cols, num_actions), dtype=float)

	steps_per_episode = np.cumsum(steps_per_episode)
	return (steps_per_episode)/runs

def Expected_sarsa( rows,  cols, num_actions, episodes_tot, wind, start, end_state, alpha, epsilon, runs, gamma, stochasticity):
	transitions = np.zeros(( rows,  cols, num_actions), dtype=float)
	steps_per_episode = np.zeros(episodes_tot, dtype=float)

	for n in range(runs):
		seed = n
		np.random.seed(seed)
		for x in range(episodes_tot):
			current_state = start
			action = policy(current_state, epsilon, num_actions, transitions)
			time_steps = 0

			while(1):
				nextstate, reward = get_nextstate_reward( rows,  cols, wind, start, end_state, current_state, action, stochasticity)
				take_action = policy(nextstate, epsilon, num_actions, transitions)
				p=np.max(transitions[nextstate[0]][nextstate[1]])
				ave=np.average(transitions[nextstate[0]][nextstate[1]])
				z=(epsilon)*ave+(1-epsilon)*p

				transitions[current_state[0]][current_state[1]][action] += alpha*(reward + gamma*z - transitions[current_state[0]][current_state[1]][action])
				current_state = nextstate
				action = take_action
				time_steps += 1
				if current_state[0] == end_state[0] and current_state[1] == end_state[1]:
					break;
			steps_per_episode[x] += time_steps 

		transitions = np.zeros(( rows,  cols, num_actions), dtype=float)

	steps_per_episode = np.cumsum(steps_per_episode)
	return (steps_per_episode)/runs


if __name__ == '__main__':
	rows = 7
	cols = 10
	epsilon = 0.1
	alpha = 0.5
	episodes_tot = 190
	runs = 15
	num_actions = 4
	gamma = 1
	wind = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=int)
	start = np.asarray([3, 0], dtype=int)
	end_state = np.asarray([3, 7], dtype=int)
	episode_arr = [i for i in range(episodes_tot)]

	stochasticity = 0
	steps_per_episode1 = sarsa0( rows,  cols, num_actions, episodes_tot, wind, start, end_state, alpha, epsilon, runs, gamma, stochasticity)
	time_steps1 = np.zeros(len(steps_per_episode1)+1)
	time_steps1[1:] = steps_per_episode1
	y1 = np.zeros(len(episode_arr)+1)
	y1[1:] = episode_arr

	num_actions = 8
	steps_per_episode2 = sarsa0( rows,  cols, num_actions, episodes_tot, wind, start, end_state, alpha, epsilon, runs, gamma, stochasticity)
	time_steps2 = np.zeros(len(steps_per_episode2)+1)
	time_steps2[1:] = steps_per_episode2
	y2 = np.zeros(len(episode_arr)+1)
	y2[1:] = episode_arr

	stochasticity = 1
	steps_per_episode3 = sarsa0( rows,  cols, num_actions, episodes_tot, wind, start, end_state, alpha, epsilon, runs, gamma, stochasticity)
	time_steps3 = np.zeros(len(steps_per_episode3)+1)
	time_steps3[1:] = steps_per_episode3
	y3 = np.zeros(len(episode_arr)+1)
	y3[1:] = episode_arr
	
	num_actions = 4
	stochasticity = 0
	steps_per_episode4 = Q_learning( rows,  cols, num_actions, episodes_tot, wind, start, end_state, alpha, epsilon, runs, gamma, stochasticity)
	time_steps4 = np.zeros(len(steps_per_episode4)+1)
	time_steps4[1:] = steps_per_episode4
	y4 = np.zeros(len(episode_arr)+1)
	y4[1:] = episode_arr

	num_actions = 4
	stochasticity = 0
	steps_per_episode5 = Expected_sarsa( rows,  cols, num_actions, episodes_tot, wind, start, end_state, alpha, epsilon, runs, gamma, stochasticity)
	time_steps5 = np.zeros(len(steps_per_episode5)+1)
	time_steps5[1:] = steps_per_episode5
	y5 = np.zeros(len(episode_arr)+1)
	y5[1:] = episode_arr
	
	
	plt.figure()
	plt.plot(time_steps1, y1)
	plt.title("Baseline Windy Gridworld, alpha={}, epsilon={}".format(alpha, epsilon))
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.grid()
	plt.savefig("Baseline_windy_gridworld.png")

	plt.figure()
	plt.plot(time_steps2, y2)
	plt.title("Kings Moves Windy Gridworld, alpha={}, epsilon={}".format(alpha, epsilon))
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.grid()
	plt.savefig("Kings_moves.png")

	plt.figure()
	plt.plot(time_steps3, y3)
	plt.title("Kings Moves & stochastic Wind, alpha={}, epsilon={}".format(alpha, epsilon))
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.grid()
	plt.savefig("Kings_moves_stochatic.png")



	plt.figure()
	plt.plot(time_steps1, y1, label="Baseline Windy Gridworld")
	plt.plot(time_steps2, y2, label="With king's moves")
	plt.plot(time_steps3, y3, label="King's moves & stochastic wind")
	plt.title("Windy Gridworld, alpha={}, epsilon={}".format(alpha, epsilon))
	plt.legend(loc='lower right')
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.grid()
	plt.savefig("Baseline_kings_stocastic.png")


	plt.figure()
	plt.plot(time_steps1, y1, label="Sarsa")
	plt.plot(time_steps5, y5, label="Expected Sarsa")
	plt.plot(time_steps4, y4, label="Q-Learning")
	plt.title("Windy Gridworld with 4 moves-Comparison, alpha={}, epsilon={}".format(alpha, epsilon))
	plt.legend(loc='lower right')
	plt.xlabel("Time Steps")
	plt.ylabel("Episodes")
	plt.grid()
	plt.savefig("Sarsa_expectedsarsa_QL.png")
