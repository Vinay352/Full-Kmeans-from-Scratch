import math

import pandas as pd
import random
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time


def readDataframe(path):
    df = pd.read_csv(path)
    return df




def createRandomKMeanValues(k, cols):
    comOfClusters = []
    for i in range(k):
        randomMean = []
        for j in range(cols):
            randomMean.append(round(random.uniform(0, 10), 3))
        comOfClusters.append(randomMean)

    return comOfClusters


def calculateDifferenceBetweenMeanAndRow(data, mean):
    temp = np.subtract(data, mean)

    i = 0
    for i in range(len(temp)):
        temp[i] = abs(temp[i])


"""
A function to calculate COM (Center of Mass) of a vector.
"""
def calculateCOM(sample, numberOfFeature):
    if isinstance(sample[0], list) is False and len(sample) == numberOfFeature : # single point
        return sample

    # if cluster - continue
    # for eg, [[0.35, 0.32], [[0.28, 0.33]], [[0.45, 0.3]], [[0.26, 0.19]]]

    sampleString = '' + str(sample) # convert list of samples into string

    # regex to find only decimals from the sampleString
    decimalsOnly = re.findall('\d*\.?\d+', sampleString)

    storeValues = [] # a temporary list to hold all values belonging to 1 vector
    finalArrayClusterPoints = [] # a list to hold all vectors with their feature values filled using the above regex

    # traverse the whole list of decimals obtained from the regex operation
    for i in range(1, len(decimalsOnly) + 1):
        # if number of features in dataset = 5,
        # then store first 5 decimal points to the first vector's list
        storeValues.append(float(decimalsOnly[i - 1]))

        # When found enough decimal points (= number of features),
        # input that into final vector list.
        # Each entry in the final vector list denotes every row of the actual dataset
        # with every decimal value of that vector's list being the column values in the dataset
        if i % numberOfFeature == 0:
            finalArrayClusterPoints.append(storeValues)
            storeValues = [] # re-initialize the list for the next vector

    numberOfItemsInCluster = len(finalArrayClusterPoints) # count of vectors total

    finalArrayClusterPoints = np.array(finalArrayClusterPoints, dtype = float)

    # Sum all vectors (add values of all data points for the same feature/column in dataset) in the final vector list
    # Add first value (column 1) of all vectors and put that as first entry in new numpy array
    # Add second value (column 2) of all vectors and put that as second entry in new numpy array
    # and so on.
    COMClusterPoints = np.sum(finalArrayClusterPoints, axis=0)

    # normalize by number of vectors
    COMClusterPoints = COMClusterPoints / numberOfItemsInCluster

    COMClusterPoints = np.round(COMClusterPoints, 3)

    return COMClusterPoints.tolist()



def kMeanAlgorithm(comOfClusters, df, k, length, cols):

    newCluster = [[] for i in range(k)]

    input = df.to_numpy(dtype = float)

    belongToCluster = []

    rowCount = 0
    while rowCount < length:
        leastDistance = float('inf')
        correspondingMeanIndex = -1
        dist = []

        for mean in comOfClusters:
            # temp = calculateDifferenceBetweenMeanAndRow(input[rowCount], mean)
            # print(input[rowCount])
            # print(mean)
            # dist = np.append(dist,np.linalg.norm(input[rowCount] - mean,axis=0).reshape(-1,1),axis=0)

            dist.append(np.linalg.norm(input[rowCount] - mean, axis=0).reshape(-1, 1).tolist()) # later change to axis = 1
            # print(dist)

        leastDistance = min(dist)
        correspondingMeanIndex = dist.index(leastDistance)
        # print(leastDistance)
        # print(correspondingMeanIndex)

        belongToCluster.append(correspondingMeanIndex) # which cluster does the row belong to (closer to)
        # print("---")

        newCluster[correspondingMeanIndex].append(input[rowCount].tolist()) # row is appended to the index whose value = which cluster number it is
        # print(newCluster)

        rowCount += 1

    comOfClusters = []
    for cluster in newCluster:
        if len(cluster) != 0:
            comOfClusters.append(calculateCOM(cluster, cols))

    return comOfClusters, belongToCluster


def elbowMethodForValueOfK(df):
    cost = []
    for i in range(1, 11):
        KM = KMeans(n_clusters=i, max_iter=500)
        KM.fit(df)

        # calculates squared error
        # for the clustered points
        cost.append(KM.inertia_)

        # plot the cost against K values
    plt.plot(range(1, 11), cost, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.ylabel("Squared Error (Cost)")
    plt.show()  # clear the plot


def calculateSSE(input, newComOfClusters, belongToCluster):
    sse = 0
    rowCount = 0
    while rowCount < len(input):
        temp = np.square( (np.linalg.norm(input[rowCount] - newComOfClusters[belongToCluster[rowCount]], axis=0).reshape(-1, 1).tolist()) )
        # print(temp[0][0])

        sse += temp[0][0]

        rowCount += 1

    return sse


def fullKMeansAlgorithm(df):
    length = len(df)  # no of rows
    cols = df.shape[1]  # number of features

    # below lines are for the elbow method to find optimal value of k
    # elbowMethodForValueOfK(df)

    # k = int(math.sqrt(length))  # select k
    k = 4

    minSSE = float('inf')
    minComOfClusters = []
    minBelongToCluster = []

    for iteration in range(10000):

        comOfClusters = createRandomKMeanValues(k, cols)  # initiate random k mean values

        # comOfClusters = [[4, 2],[12, 11],[20, 25]]

        # print(comOfClusters)

        newComOfClusters = []  # new centers of clusters
        belongToCluster = []  # indices represent which cluster, and the list at that index represent which all data points are in that cluster
        while True:
            newComOfClusters, belongToCluster = kMeanAlgorithm(comOfClusters, df, k, length, cols)
            if np.array_equal(newComOfClusters, comOfClusters):
                break
            else:
                comOfClusters = newComOfClusters

        if iteration % 100 == 0:
            print(iteration)

        # newComOfClusters = newComOfClusters.tolist()
        # print(newComOfClusters)
        # print(len(newComOfClusters))
        # print(belongToCluster)
        # print(len(belongToCluster))

        tempSSE = calculateSSE(df.to_numpy(dtype=float), newComOfClusters, belongToCluster)

        # print(tempSSE)

        if minSSE >= tempSSE:
            minSSE = tempSSE
            minComOfClusters = newComOfClusters
            minBelongToCluster = belongToCluster

    return minSSE, minComOfClusters, minBelongToCluster


def main():
    path = "HW_CLUSTERING_SHOPPING_CART_v2221A.csv"
    # path = "test.csv"
    df = readDataframe(path)
    # print(df)

    df = df[['  Milk', 'ChildBby', 'Vegges', 'Cereal', ' Bread', '  Rice',
             '  Meat', '  Eggs', 'YogChs', ' Chips', '  Soda', ' Fruit', '  Corn',
             '  Fish', ' Sauce', ' Beans', 'Tortya', '  Salt', 'Scented', ' Salza']]

    start_time = time.time()

    minSSE, minComOfClusters, minBelongToCluster = fullKMeansAlgorithm(df)

    end_time = time.time()

    print("time taken = " + str(end_time - start_time))

    print(minSSE)
    print(minComOfClusters)
    print(len(minComOfClusters))
    print(minBelongToCluster)
    print(len(minBelongToCluster))


# main conditional guard
if __name__ == "__main__":
    main()