#!/usr/bin/python3

def knn_improved(data, k_neighbors, input):
    data_info = []
    neighbors = []
    for entry in data:
        s = {'classification': entry[0], 'distance': dist(input, entry)}
        data_info.append(s)

    data_info = sorted(data_info, key = lambda i: i['distance'])

    for entry in data_info[:k_neighbors]:
        neighbors.append(entry['classification'])

    return mode(neighbors)

# this function was left here just to show the base algorithm
def knn(data, k_neighbors, input):
    neighbors = []
    classifiers = []

    for entry in data:
        if len(neighbors) < k_neighbors:
            neighbors.append(entry)
        else:
            new_dist = dist(input, entry)

            for neighbor in neighbors:
                if new_dist < dist(input, neighbor):
                    neighbors.remove(neighbor)
                    neighbors.append(entry)

    for neighbor in neighbors:
        classifiers.append(neighbor[0])

    return mode(classifiers)


def dist(base, entry):
    dist = 0

    for i in range(len(base)):
        dist = (int(base[i]) + int(entry[i])) ** 2

    return dist ** 0.5

def mode(entries):
    classifiers = {}

    for entry in entries:
        if entry in classifiers:
            classifiers[entry] += 1
        else:
            classifiers[entry] = 1

    return max(classifiers, key = classifiers.get)

if __name__ == "__main__":
    filename = '/Users/gattlinwalker/Documents/school_work/MSU/csc/csc535/Data-Mining/CSC535-HW2/test/MNIST_train.csv'
    data = []

    with open(filename, 'r') as file:
        data = [line.strip('\n').split(',') for line in file.readlines()]
        data.pop(0) # removing the first entry of column names

    # This is for the base implementation
    # results = knn(data, 5, data[300])

    results = knn_improved(data, 5, data[300])
    print(results)