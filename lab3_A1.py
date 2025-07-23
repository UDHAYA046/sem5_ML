import numpy as np

# Sample dataset: Each row represents a sample, and the last column represents the class label
data = np.array([
    [1.0, 2.0, 0],
    [1.5, 1.8, 0],
    [1.2, 1.9, 0],
    [5.0, 8.0, 1],
    [6.0, 9.0, 1],
    [5.5, 8.5, 1]
])

# Separate the data into two classes
class_0 = data[data[:, -1] == 0][:, :-1]  # Features of class 0
class_1 = data[data[:, -1] == 1][:, :-1]  # Features of class 1

# Calculate the mean (centroid) for each class
centroid_0 = class_0.mean(axis=0)
centroid_1 = class_1.mean(axis=0)

# Calculate the spread (standard deviation) for each class
spread_0 = class_0.std(axis=0)
spread_1 = class_1.std(axis=0)

# Calculate the distance between the mean vectors of the two classes
interclass_distance = np.linalg.norm(centroid_0 - centroid_1)

# Print the results
print("Class 0 Centroid:", centroid_0)
print("Class 0 Spread (Standard Deviation):", spread_0)
print("Class 1 Centroid:", centroid_1)
print("Class 1 Spread (Standard Deviation):", spread_1)
print("Interclass Distance:", interclass_distance)
