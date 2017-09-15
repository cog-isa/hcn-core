import random
import copy

class DSU:
    def __init__(self, number_of_sets):
        self.__parent = [i for i in range(number_of_sets)]
        self.__rank = [0] * number_of_sets

    def add_set(self):
        self.__parent.append(len(self.__parent))
        self.__rank.append(0)

    def find_set(self, num):
        if num == self.__parent[num]:
            return num
        self.__parent[num] = self.find_set(self.__parent[num])
        return self.__parent[num]

    def union_sets(self, num1, num2):
        v1 = self.find_set(num1)
        v2 = self.find_set(num2)
        if (v1 != v2):
            if (self.__rank[v1] < self.__rank[v2]):
                v1, v2 = v2, v1
            self.__parent[v2] = v1
            if self.__rank[v1] == self.__rank[v2]:
                self.__rank[v1] += 1


def hierarchical(dist_matrix, max_clusters = 1):
    num_elems = len(dist_matrix)
    dist_line = []
    for i in range(num_elems):
        for j in range(num_elems):
            dist_line.append(tuple([dist_matrix[i][j], i, j]))

    dist_line.sort()
    num_clusters = num_elems
    dsu = DSU(num_elems)
    ind = 0
    while num_clusters > max_clusters:
        cur_elem = dist_line[ind]
        (v1, v2) = cur_elem[1], cur_elem[2]
        if (dsu.find_set(v1) != dsu.find_set(v2)):
            num_clusters -= 1
            dsu.union_sets(v1, v2)
        ind += 1

    cluster_index = [dsu.find_set(i) for i in range(num_elems)]
    scale_numbers = list(set(cluster_index))
    scale_cluster_index = [scale_numbers.index(cluster_index[i]) for i in range(num_elems)]
    return scale_cluster_index


class OnlineKmeansClusterer:
    def __init__(self, k, dimension, max_val):
        self.__k = k
        self.__dimension = dimension
        self.__centers = [[random.randint(0, max_val) for j in range(dimension)] for i in range(k)]
        self.__num_elems = [0] * k

    def get_centers(self):
        return copy.deepcopy(self.__centers)

    def calc_distance(self, pattern, center):
        dist = 0
        for i in range(self.__dimension):
            dist += (pattern[i] - center[i]) ** 2
        return dist ** 0.5

    def update_clusterer(self, pattern):
        MAXVAL = 1e10
        min_dist = MAXVAL
        cluster = 0

        for i in range(self.__k):
            cur_dist = self.calc_distance(pattern, self.__centers[i])
            if (cur_dist < min_dist):
                min_dist = cur_dist
                cluster = i

        self.__num_elems[cluster] += 1
        # move center
        for i in range(self.__dimension):
            self.__centers[cluster][i] += (pattern[i] - self.__centers[cluster][i]) / self.__num_elems[cluster]
        return cluster
