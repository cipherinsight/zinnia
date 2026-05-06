# Source: Pythran tests/cases/kmeans.py
# Original #pythran export: test()
# Migration notes: relies on random.sample inside the function; likely unsupported.
from zinnia import *
import math
import random
from functools import reduce


@zk_chip
def calculateCentroid(cluster) -> List[Float]:
    reduce_coord = lambda i: reduce(lambda x, p: x + p[i], cluster, 0.0)
    centroid_coords = [reduce_coord(i) / len(cluster) for i in range(len(cluster[0]))]
    return centroid_coords


@zk_chip
def getDistance(a, b) -> Float:
    ret = reduce(lambda x, y: x + pow((a[y] - b[y]), 2), range(len(a)), 0.0)
    return math.sqrt(ret)


@zk_chip
def makeRandomPoint(n, lower, upper) -> List[Float]:
    return [random.uniform(lower, upper) for i in range(n)]


@zk_chip
def kmeans(points, k, cutoff) -> List[List[Float]]:
    initial = random.sample(points, k)
    clusters = [[p] for p in initial]
    centroids = [calculateCentroid(c) for c in clusters]
    while True:
        lists = [[] for c in clusters]
        for p in points:
            smallest_distance = getDistance(p, centroids[0])
            index = 0
            for i in range(len(clusters[1:])):
                distance = getDistance(p, centroids[i + 1])
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i + 1
            lists[index].append(p)
        biggest_shift = 0.0
        for i in range(len(clusters)):
            if lists[i]:
                new_cluster, new_centroid = (lists[i], calculateCentroid(lists[i]))
                shift = getDistance(centroids[i], new_centroid)
                clusters[i] = new_cluster
                centroids[i] = new_centroid
                biggest_shift = max(biggest_shift, shift)
        if biggest_shift < cutoff:
            break
    return clusters


@zk_circuit
def test():
    num_points, dim, k, cutoff, lower, upper = 500, 10, 50, 0.001, 0, 2000
    points = [makeRandomPoint(dim, lower, upper) for i in range(num_points)]
    clusters = kmeans(points, k, cutoff)
    _zinnia_result = clusters
