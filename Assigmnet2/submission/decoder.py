import sys
import numpy as np

f = open(sys.argv[2],"r")	
lines = f.readlines()

g = open(sys.argv[4],"r")	
vals = g.readlines()

p = 1.0 

maze = []
for line in lines:
	rows = line.strip().split()
	maze.append(rows)

values = vals[:-1]
values = [x.strip().split() for x in values]
#Assign_direction
direction = {}
direction[0] = 'N'
direction[1] = 'E'
direction[2] = 'S'
direction[3] = 'W'

# def the variables
state = 0 
actions = 4 
rows=len(maze)
cols=len(maze[0])
end=[]
states = np.array([[-1]*cols]*rows, dtype=np.int64)


for x in range(rows):
	for y in range(cols):
		if maze[x][y] == '1':
			continue
		if maze[x][y] == '2':
			start = state
			cordx = x 
			cordy = y
		if maze[x][y] == '3':
			end.append(state)	
		states[x][y] = state
		state += 1 


actions = []
while start not in end:
	act = int(values[start][1])
	actions.append(act)
	cordx = cordx - pow( - 1 , act // 2) * (1 - (act % 2))
	cordy = cordy + pow( - 1 , act // 2) * (act % 2)
	start = states[cordx][cordy]

actions = list(map(lambda x: direction[x] , actions))
"""
Print the path in terms of directions
"""
print(" ".join(actions))


