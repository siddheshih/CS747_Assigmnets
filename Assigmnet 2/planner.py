import numpy as np 
import sys
from pulp import LpMinimize, LpProblem, lpSum, LpVariable,LpSolverDefault
import pulp
#import time

#begin=time.time()

def bellman_equations(Trans, Rew, Val, gamma):
    numStates = np.size(Trans)
    Trans = np.reshape(Trans, (numStates, 1))
    Rew = np.reshape(Rew, (numStates, 1))
    Rew = np.add(Rew , gamma * Val)
    result = np.matmul(np.transpose(Trans), Rew)
    return result[0][0]

def valiter(numStates,numActions,start,end,rewards,prob_dict,mdtype,discount):
	#policy=np.ones([numStates, numActions])
    oldValue = np.zeros(numStates)
    nonTerminals = [st for st in range(numStates) if st not in end]
    policy=np.ones([numStates, numActions])
    actions = np.zeros(numStates)
    tol = 1e-20
    maxIter = 100000
    t = 0
    while(t<maxIter):
        newValue = np.zeros(numStates)
        for i in prob_dict:
            if i not in end:
                maxScore = -float('inf')
                bestAction = -1
                for j in prob_dict[i]:
                    transValue = 0
                    for trans in prob_dict[i][j]:
                        transValue += trans[2]*(trans[1]+discount*oldValue[trans[0]])
                    if transValue > maxScore:
                        maxScore = transValue
                        bestAction = j
                actions[i] = bestAction
                newValue[i] = maxScore
        if np.sum(abs(oldValue-newValue))<tol: break
        #print(np.sum(abs(oldValue-newValue)))
        for i in range(numStates):
            oldValue[i] = newValue[i]
        t = t+1
    return oldValue, actions
	 

def linear_prog(numStates, numActions, rewards, transition, discount, mdpType,end):
    lin_prog = LpProblem("MDP", LpMinimize)
    decision_var = LpVariable.dict("value_function", range(numStates))
    LpSolverDefault.msg = 0
    # The Objective Function
    lin_prog += lpSum([decision_var[i] for i in range(numStates)])
    # adding constraints
    V = np.zeros((numStates, 1), dtype = LpVariable)
    for i in range(numStates):
        V[i][0] = decision_var[i]
    for state in range(numStates):
        for action in range(numActions):
            lowerBound = bellman_equations(transition[state][action], rewards[state][action], V, discount)
            lin_prog += decision_var[state] >= lowerBound
    
    if (mdpType == "episodic"):
    	for i in end :
    		lin_prog += (decision_var[i] == 0)
    
    lin_prog.solve(pulp.PULP_CBC_CMD(msg=0))
    V = np.zeros((numStates, 1))
    for i in range(numStates):
        V[i][0] = decision_var[i].varValue
    
    P = np.zeros((numStates, 1), dtype = int)
    tmp = np.zeros((numActions, ))
    for state in range(numStates):
        for action in range(numActions):
            tmp[action] = bellman_equations(transition[state][action], rewards[state][action], V, discount)
        P[state][0] = np.argmax(tmp)
    return P, V

def policyiter(numStates, numActions, rewards, transition, discount, mdpType,end):
    Policy = np.zeros((numStates, 1), dtype = int)         
    Value = np.zeros((numStates, 1))                       
    tmp = np.zeros((numActions, ))
    epsilon = 1e-10
    # random initialization of policy
    for i in range(numStates):
        Policy[i][0] = np.random.choice(range(numActions))
    while True:
        delta = 0
        old_V = np.copy(Value)
        old_P = np.copy(Policy)
        # policy evaluation
        for state in range(numStates):
            if (mdpType == "episodic" and state in end):
                Value[state][0] = 0
            else:
                p = Policy[state][0]
                Value[state][0] = bellman_equations(transition[state][p], rewards[state][p], Value, discount)
            delta = max(delta, abs(Value[state][0] - old_V[state][0]))
        if (delta < epsilon):
            # policy improvement
            stable_policy_found = True
            for state in range(numStates):
                for action in range(numActions):
                    tmp[action] = bellman_equations(transition[state][action], rewards[state][action], Value, discount)
                Policy[state][0] = np.argmax(tmp)
                if (Policy[state][0] != old_P[state][0]):
                    stable_policy_found = False
                    break
            if (stable_policy_found):
                break
    return Policy, Value


if __name__ =="__main__":

    file=sys.argv[2]

    algo=sys.argv[4]
    states=0
    discount=0.000
    event_type=""
    rewards=[]
    prob_dict={}
    transition_p=[]

    start=0
    end=[]
    with open(file,'r') as f:
        lines=f.read().splitlines()
    for line in lines:
        token=line.split()
        first_word=token[0]
        if first_word=="numStates":
            states=int(token[1])
        elif first_word=="numActions":
            actions=int(token[1])
        elif first_word=="start":
            start=int(token[1])
            rewards = np.resize(rewards,(states,actions,states))
            transition_p = np.resize(transition_p,(states,actions,states))

        elif first_word=="end":
            if token[1]=="-1":
                end=[]
            else:
                end=[int(val) for val in token[1:]]
        elif first_word=="discount":
            discount=float(token[1])
        elif first_word=="mdptype":
            event_type=token[1]
        elif first_word=="transition":
            st = int(token[1])
            act = int(token[2])
            sNew = int(token[3])
            rew = float(token[4])
            prob = float(token[5])
            try: prob_dict[st][act].append((sNew, rew, prob))
            except:
                try: prob_dict[st][act] = [(sNew, rew, prob)]
                except: prob_dict[st] = {act: [(sNew, rew, prob)]}
            rewards[int(token[1]),int(token[2]),int(token[3])] = float(token[4])
            transition_p[int(token[1]),int(token[2]),int(token[3])] = float(token[5])
		#else:
		#	print("Please Check the inputs")

	# print(states)
	# print(actions)
	# print(start)
	# print(end)
	# print(discount)
	# print(rewards)
	# print(transition_p)
    

    if (algo =="vi"):
        [values,actions]=valiter(states,actions,start,end,rewards,prob_dict,event_type,discount)
    if (algo=="hpi"):
        [actions,values]=policyiter(states,actions,rewards,transition_p,discount,event_type,end)
    if (algo=="lp"):
        [actions,values]=linear_prog(states,actions,rewards,transition_p,discount,event_type,end)
    for i in range(states):
        print("{:.6f}".format(float(values[i]))+" "+str(int(actions[i])))
    



# end=time.time()

# print(end-begin)













