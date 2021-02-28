import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file1 = open('outputDataT1.txt','r')
file2 = open('outputDataT2.txt','r')
lines = file1.readlines()
lines2 = file2.readlines()

instances = {"../instances/i-1.txt":0,"../instances/i-2.txt":1,"../instances/i-3.txt":2}
algorithms = {"epsilon-greedy":0,"ucb":1,"kl-ucb":2,"thompson-sampling":3}
#epsilons = {"-":0,"0.002":0,"0.02":1,"0.2":2}
horizons = {"100":0,"400":1,"1600":2,"6400":3,"25600":4,"102400":5}

data = np.zeros([len(instances),len(algorithms),len(horizons)])

for line in lines:
    vals = line.split(', ')
    #print(vals[1])
    data[instances[vals[0]],algorithms[vals[1]],horizons[vals[4]]] += float(vals[5]) / 50.0

#instance 1
x = np.log10(np.asarray([100, 400, 1600, 6400, 25600, 102400]))
y1 = data[0,0,:]
y2 = data[0,1,:]
y3 = data[0,2,:]
y4 = data[0,3,:]
plt.title(' Plot for instance-1, Task-1')
plt.xscale('log')
plt.plot(x,y1, marker='o', markerfacecolor='#F1C40F', markersize=6, color='#AF7AC5', linewidth=2.5, label="epsilon-greedy (with epsilon=0.02)",alpha=0.7)
plt.plot(x,y2, marker='o', markerfacecolor='#9B59B6', markersize=6, color='#F7DC6F', linewidth=2.5, label="ucb",alpha=0.7)
plt.plot(x,y3, marker='o', markerfacecolor='#E67E22', markersize=6, color='#EB984E', linewidth=2.5, label="kl-ucb",alpha=0.7)
plt.plot(x,y4, marker='o', markerfacecolor='#34495E', markersize=6, color='#5D6D7E', linewidth=2.5, label="thompson-sampling",alpha=0.7)
plt.legend()
plt.xlabel("log(Horizon)")
plt.ylabel("Regret")
# plt.show()
plt.savefig('instance1.png',bbox_inches='tight')
plt.clf()

# instance 2
x = np.log10(np.asarray([100, 400, 1600, 6400, 25600, 102400]))
y1 = data[1,0,:]
y2 = data[1,1,:]
y3 = data[1,2,:]
y4 = data[1,3,:]
plt.title(' Plot for instance-2, Task-1')
plt.xscale('log')
plt.plot(x,y1, marker='o', markerfacecolor='#3498DB', markersize=5, color='#5DADE2', linewidth=2.5, label="epsilon-greedy (with epsilon 0.02)",alpha=0.7)
# plt.plot(x,y4, marker='o', markerfacecolor='#27AE60', markersize=5, color='#52BE80', linewidth=2, label="epsilon-greedy (0.2)",alpha=0.7)
plt.plot(x,y2, marker='o', markerfacecolor='#F1C40F', markersize=5, color='#F7DC6F', linewidth=2.5, label="ucb",alpha=0.7)
plt.plot(x,y3, marker='o', markerfacecolor='#E67E22', markersize=5, color='#EB984E', linewidth=2.5, label="kl-ucb",alpha=0.7)
plt.plot(x,y4, marker='o', markerfacecolor='#34495E', markersize=5, color='#5D6D7E', linewidth=2.5, label="thompson-sampling",alpha=0.7)
plt.xlabel("log(Horizon)")
plt.ylabel("Regret")
plt.legend()
# plt.show()
plt.savefig('instance2.png',bbox_inches='tight')
plt.clf()

# instance 3
x = np.log10(np.asarray([100, 400, 1600, 6400, 25600, 102400]))
y1 = data[2,0,:]
y2 = data[2,1,:]
y3 = data[2,2,:]
y4 = data[2,3,:]
plt.title(' Plot for instance-3, Task-1')
plt.xscale('log')
plt.plot(x,y1, marker='o', markerfacecolor='#3498DB', markersize=5, color='#5DADE2', linewidth=2.5, label="epsilon-greedy (with epsilon 0.02)",alpha=0.7)
# plt.plot(x,y4, marker='o', markerfacecolor='#27AE60', markersize=5, color='#52BE80', linewidth=2, label="epsilon-greedy (0.2)",alpha=0.7)
plt.plot(x,y2, marker='o', markerfacecolor='#F1C40F', markersize=5, color='#F7DC6F', linewidth=2.5, label="ucb",alpha=0.7)
plt.plot(x,y3, marker='o', markerfacecolor='#E67E22', markersize=5, color='#EB984E', linewidth=2.5, label="kl-ucb",alpha=0.7)
plt.plot(x,y4, marker='o', markerfacecolor='#34495E', markersize=5, color='#5D6D7E', linewidth=2.5, label="thompson-sampling",alpha=0.7)
plt.xlabel("log(Horizon)")
plt.legend()
plt.ylabel("Regret")
# plt.show()
plt.savefig('instance3.png',bbox_inches='tight')
plt.clf()


### T2 #####

algorithms2 = {"thompson-sampling":0, "thompson-sampling-with-hint":1}

data2 = np.zeros([len(instances),len(algorithms2),len(horizons)])
for line in lines2:
    vals = line.split(', ')
    #print(vals[1])
    data2[instances[vals[0]],algorithms2[vals[1]],horizons[vals[4]]] += float(vals[5]) /50

print(data2)

# instance 1
x = np.log10(np.asarray([100, 400, 1600, 6400, 25600, 102400]))
y1 = data2[0,0,:]
y2 = data2[0,1,:]
plt.title(' Plot for instance-1, Task-2')
plt.xscale('log')
plt.plot(x,y1, marker='o', markerfacecolor='#9B59B6', markersize=5, color='#AF7AC5', linewidth=2.5, label="thompson-sampling",alpha=0.7)
plt.plot(x,y2, marker='o', markerfacecolor='#F1C40F', markersize=5, color='#F7DC6F', linewidth=2.5, label="thompson-sampling-with-hint",alpha=0.7)
plt.legend()
plt.ylabel("Regret")
plt.xlabel("log(Horizon)")
# plt.show()
plt.savefig('instance1_T2.png',bbox_inches='tight')
plt.clf()

# instance 2
x = np.log10(np.asarray([100, 400, 1600, 6400, 25600, 102400]))
y1 = data2[1,0,:]
y2 = data2[1,1,:]
plt.title(' Plot for instance-2, Task-2')
plt.xscale('log')
plt.plot(x,y1, marker='o', markerfacecolor='#9B59B6', markersize=5, color='#AF7AC5', linewidth=2.5, label="thompson-sampling",alpha=0.7)
plt.plot(x,y2, marker='o', markerfacecolor='#F1C40F', markersize=5, color='#F7DC6F', linewidth=2.5, label="thompson-sampling-with-hint",alpha=0.7)
plt.legend()
plt.xlabel("log(Horizon)")
plt.ylabel("Regret")
# plt.show()
plt.savefig('instance2_T2.png',bbox_inches='tight')
plt.clf()

# instance 3
x = np.log10(np.asarray([100, 400, 1600, 6400, 25600, 102400]))
y1 = data2[2,0,:]
y2 = data2[2,1,:]
plt.title(' Plot for instance-3, Task-2')
plt.xscale('log')
plt.plot(x,y1, marker='o', markerfacecolor='#9B59B6', markersize=5, color='#AF7AC5', linewidth=2.5, label="thompson-sampling",alpha=0.7)
plt.plot(x,y2, marker='o', markerfacecolor='#F1C40F', markersize=5, color='#F7DC6F', linewidth=2.5, label="thompson-sampling-with-hint",alpha=0.7)
plt.legend()
plt.xlabel("log(Horizon)")
plt.ylabel("Regret")
# plt.show()
plt.savefig('instance3_T2.png',bbox_inches='tight')
plt.clf()