import sys
import numpy as np

if __name__ =="__main__":
	f=open(sys.argv[2],'r')
	lines=f.readlines()
	maze=[]
	for line in lines:
		row=line.strip().split()
		maze.append(row)
	p=1.0
	state=0
	actions=4
	rows=len(maze)
	cols=len(maze[0])
	end=[]
	states = np.array([[-1]*cols]*rows, dtype=np.int64)
	for x in range(rows):
		for y in range(cols):
			if maze[x][y]=="1":
				continue
			if maze[x][y]=="2":
				st_start=state
			if maze[x][y]=="3":
				end.append(state)
			states[x][y]=state
			state=state+1
	transitions={}
	for x in range(rows):
		for y in range(cols):
			if maze[x][y] == '1' or maze[x][y] == '3':
				continue
			N=maze[x-1][y]
			E=maze[x][y+1]
			S=maze[x+1][y]
			W=maze[x][y-1]
			validM = (N != '1') + (E != '1') + (S != '1') + (W != '1')
			validL = [N , E  , S , W]
			state_val = [states[x - 1][y] , states[x][y + 1] , states[x + 1][y], states[x][y - 1]]
			if validM == 0:
				validM=1
			transp=1
			
			s = states[x][y]
			for i in range(4):
				if validL[i] == '1':
					transitions[s , i , s] = [ -1, 1]
					continue
				for j in range(4):
					reward=-1
					if validL[j] == '3':
						reward = 1000 + state
					if (i==j):
						transitions[s , i , state_val[i]] = [reward , transp]
					if state_val[j] == -1:
						continue
	
	print("numStates" , state)
	print("numActions" , 4)
	print("start",st_start)
	print("end" , " ".join(list(map( str , end))))
	for t in transitions:
		print("transition" , t[0] , t[1] , t[2]  , transitions[t][0] , transitions[t][1])
	print("discount" , 0.9999)





 






