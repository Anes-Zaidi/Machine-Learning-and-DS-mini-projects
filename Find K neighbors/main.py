import numpy as np
import matplotlib.pyplot as plt
import random

points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
k = 3

def kNearestNighboor(points , k):
    points = np.array(points)
    results = []
    for i in range(len(points)):
        distances = np.sqrt(np.sum((points[i] - points) ** 2 , axis = 1))
        nearestDist = np.partition(distances , k)[:k]
        nearstPointsIndcs = np.argpartition(distances , k)[:k]
        results.append(points[nearstPointsIndcs])
    return results

x ,y = zip(*points)
neighbArr = kNearestNighboor(points , k)
neighbors =  [tuple(point) for sublist in neighbArr for point in sublist]
i , j  = zip(*neighbors)

plt.style.use('ggplot') 
plt.scatter(x , y , s=100)
plt.plot(i , j , color ="black")

plt.show()

