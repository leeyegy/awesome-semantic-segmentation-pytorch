import numpy as np 
data = np.loadtxt("road_target.txt")
print(data.shape)
print(data)

for i in range(480):
    for j in range(480):
        if data[i][j] >=6.0 and data[i][j]<=8:
            print(data[i][j])
