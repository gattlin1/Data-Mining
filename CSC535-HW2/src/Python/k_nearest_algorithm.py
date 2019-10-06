"""
    author: Gattlin Walker
    email: gattlin1@live.missouristate.edu
    trace: gat1

    An implementation of the K Nearest Neighbors Algorithm
    The data files are in a specific format with the first row being column names
    and the first entry in subsequent rows being the expected classification.
"""

"""
    a classification algorithm that finds the k nearest entries to the input
    @param data: the dataset used to classify the input
    @param k_neighbors: an int of the closest neighbors to consider as possible classes
    @param input: the entry to be classified
    @return: the calculated classification
"""
def knn(data, k_neighbors, input):
    data_info, neighbors = [], []

    for entry in data:
        entry_data = {'class': entry[0], 'distance': dist(input[1:], entry[1:])}
        data_info.append(entry_data)

    data_info.sort(key = lambda i: i['distance'])

    neighbors = data_info[: k_neighbors]
    closest_match = closest(neighbors)

    return closest_match


"""
    euclidean distance method to get the distance between two entries
    @param base: the entry we are trying to find the classification for
    @param entry: the entry we are comparing to the base
    @return: returns the distance of the two params
"""
def dist(base, entry):
    dist = 0

    for i in range(len(base)):
        dist += (int(base[i]) - int(entry[i])) ** 2

    return dist ** 0.5


"""
    takes an array of dictionaries and adds their class value
    to a dictionary where the value is the inverse of their distance
    from the input. The highest value is then chosen as the closest input.
    @param entries: the k nearest entries to the input
    @return: the closest entry to the input
"""
def closest(entries):
    classifiers = {}

    for entry in entries:
        classification = entry['class']
        dist = 1 / int(entry['distance']) if int(entry['distance']) > 0 else 0

        if classification in classifiers:
            classifiers[classification] += dist
        else:
            classifiers[classification] = dist

    return max(classifiers, key = classifiers.get)


"""
    takes a file path to a dataset in csv form and converts it into a dataset
    @param file_path: file path to the dataset to be made
    @return: a list of lists to form the finished dataset
"""
def make_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        dataset = [line.strip('\n').split(',') for line in file.readlines()]
    dataset.pop(0) # removing the first entry of column names

    return dataset


if __name__ == "__main__":
    training_path = '../../test/MNIST_train.csv'
    testing_path = '../../test/MNIST_test.csv'
    training_set = make_dataset(training_path)
    testing_set = make_dataset(testing_path)
    num_neighbors = 7
    correct_class = 0

    print('K = ', num_neighbors)
    for entry in testing_set:
        actual = knn(training_set, num_neighbors, entry)
        expected = entry[0]

        if expected == actual:
            correct_class += 1

        print('Desired Class: ', expected, ' Computed Class: ', actual)

    percentage_correct = 100 * correct_class / len(testing_set)

    print('Accuracy Rate: ', percentage_correct, '%')
    print('Number of misclassified test samples: ', len(testing_set) - correct_class)
    print('Total number of test samples: ', len(testing_set))
