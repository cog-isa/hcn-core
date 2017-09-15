from math import exp
import clustering


class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def add(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def radd(self, other):
        return Point(self.x + other.x, self.y + other.y)


class Node:
    def __init__(self, point_s = Point(), point_f = Point(), num_groups = 4, sigma2 = 5):
        self.start_point = point_s    # connection
        self.finish_point = point_f   # area
        self.sigma2 = sigma2          # sigma^2
        self.INF = 1e10

        self.prev_pattern_index = None
        self.number_patterns = 0
        self.pattern_matrix = []

        self.Markov_graph = []
        self.normalized_Markov_graph = []
        self.distance_graph = []      # inversely to normalized graph

        self.temporal_groups = []
        self.num_groups = num_groups

        self.output = []              # request result

    def extract_pattern(self, prev_layer):
        pattern = []
        node_matrix = prev_layer.node_matrix
        for i in range(self.start_point.x, self.finish_point.x):
            for j in range(self.start_point.y, self.finish_point.y):
                cur_line = node_matrix[i][j].get_output()
                max_value = max(cur_line)
                if max_value > 0:
                    cur_line = [1 *(cur_line[i] == max_value) for i in range(len(cur_line))]
                pattern += cur_line[:]
                ##pattern += node_matrix[i][j].get_output()
        return pattern

    def add_pattern(self, pattern):
        if pattern not in self.pattern_matrix:
            index = self.number_patterns
            self.pattern_matrix.append(pattern)
            self.number_patterns += 1
            for i in range(self.number_patterns - 1):
                self.Markov_graph[i] += [0]
                self.normalized_Markov_graph[i] += [0]
                self.distance_graph[i] += [self.INF]

            self.Markov_graph.append([0] * self.number_patterns)
            self.normalized_Markov_graph.append([0] * self.number_patterns)
            self.distance_graph.append([self.INF] * self.number_patterns)
        else:
            index = self.pattern_matrix.index(pattern)

        prev_index = self.prev_pattern_index
        if(prev_index != None):
            self.Markov_graph[prev_index][index] += 1

        self.prev_pattern_index = index

    def normalize_graph(self):
        for i in range(self.number_patterns):
            number_links = sum(self.Markov_graph[i])
            for j in range(self.number_patterns):
                if (number_links == 0):
                    continue
                self.normalized_Markov_graph[i][j] = self.Markov_graph[i][j] / number_links

    def recalc_distance(self):
        for i in range(self.number_patterns):
            for j in range(self.number_patterns):
                if self.normalized_Markov_graph[i][j] == 0:
                    self.distance_graph[i][j] = self.INF
                else:
                    self.distance_graph[i][j] = 1 / self.normalized_Markov_graph[i][j]

    def partition_temporal(self):
        if len(self.normalized_Markov_graph) == 1:
            self.temporal_groups = [0]
            return
        self.temporal_groups = clustering.hierarchical(self.distance_graph, self.num_groups)

    def get_output(self):
        return self.output[:]

    def set_output_data(self, output):
        self.output = output[:]

    def calculate(self, prev_layer, study_mode):
        pattern = self.extract_pattern(prev_layer)
        if study_mode:
            self.add_pattern(pattern)
            self.normalize_graph()
            self.recalc_distance()
            self.partition_temporal()
            return
        # in sensing node
        pattern_len = len(pattern)
        dist_square = [0] * self.number_patterns
        probability = [0] * self.num_groups

        for i in range(self.number_patterns):
            value = 0
            temp_group = self.temporal_groups[i]  # temp_group >= 1
            temp_group -= 1
            cur_pattern = self.pattern_matrix[i]
            for j in range(pattern_len):
                value += (cur_pattern[j] - pattern[j]) ** 2

            dist_square[i] = value
            new_prob = exp(-dist_square[i] / self.sigma2)
            probability[temp_group] = max(new_prob, probability[temp_group])

        summ_prob = sum(probability)
        for i in range(len(probability)):
            probability[i] /= summ_prob

        self.set_output_data(probability)

### Dealing with noise

class ClusteredNode(Node):
    def __init__(self, point_s = Point(), point_f = Point(),
             num_groups = 4, sigma2 = 5, layer_index = 0,
             gamma = 1, kmeans_k = 5, dimension = 1, max_val = 1):
        super().__init__(point_s, point_f, num_groups, sigma2)

        self.layer_index = layer_index
        self.gamma = gamma

        if (layer_index == 1):
            self.Clusterer = clustering.OnlineKmeansClusterer(kmeans_k, dimension, max_val)
            self.number_patterns = kmeans_k
            self.pattern_matrix = self.Clusterer.get_centers()

            self.Markov_graph = [[0] * kmeans_k for i in range(kmeans_k)]
            self.normalized_Markov_graph = [[0] * kmeans_k for i in range(kmeans_k)]
            self.distance_graph = [[self.INF] * kmeans_k for i in range(kmeans_k)]

    def extract_pattern(self, prev_layer):
        pattern = []
        node_matrix = prev_layer.node_matrix
        for i in range(self.start_point.x, self.finish_point.x):
            for j in range(self.start_point.y, self.finish_point.y):
                cur_line = node_matrix[i][j].get_output()
                if self.layer_index > 1:
                    max_value = max(cur_line)
                    if max_value > 0:   # extract winners -> 1, other -> 0
                        cur_line = [1 *(cur_line[i] == max_value)
                                    for i in range(len(cur_line))]
                pattern += cur_line[:]
        return pattern

    def extract_probability_vector(self, prev_layer):
        prob_vector = []
        node_matrix = prev_layer.node_matrix
        for i in range(self.start_point.x, self.finish_point.x):
            for j in range(self.start_point.y, self.finish_point.y):
                cur_line = node_matrix[i][j].get_output()
                prob_vector += cur_line[:]
        return prob_vector

    def preclustering(self, pattern):   # find, update and return the closest cluster center
        cluster_index = self.Clusterer.update_clusterer(pattern)
        self.pattern_matrix = self.Clusterer.get_centers()
        return cluster_index

    def add_pattern(self, pattern):
        if (self.layer_index > 1):
            super().add_pattern(pattern)
            return
        # if layer_index == 1
        index = self.preclustering(pattern)
        prev_index = self.prev_pattern_index

        if(prev_index != None):
            self.Markov_graph[prev_index][index] += 1

        self.prev_pattern_index = index

    def calculate(self, prev_layer, study_mode):
        pattern = self.extract_pattern(prev_layer)
        if study_mode:
            self.add_pattern(pattern)
            self.normalize_graph()
            self.recalc_distance()
            self.partition_temporal()
            return
        # in sensing node
        pattern_len = len(pattern)
        dist_square = [0] * self.number_patterns
        probability = [0] * self.num_groups

        if self.layer_index == 1:
            for i in range(self.number_patterns):
                value = 0
                temp_group = self.temporal_groups[i]  # temp_group >= 1
                temp_group -= 1
                cur_pattern = self.pattern_matrix[i]
                for j in range(pattern_len):
                    value += (cur_pattern[j] - pattern[j]) ** 2

                dist_square[i] = value
                new_prob = exp(-dist_square[i] / self.sigma2)
                probability[temp_group] = max(new_prob, probability[temp_group])
        else:
        # sensing for higher-level (>= 2) nodes
            prob_input_pattern_vector = self.extract_probability_vector(prev_layer)
            for i in range(self.number_patterns):
                temp_group = self.temporal_groups[i]  # temp_group >= 1
                temp_group -= 1
                cur_pattern = self.pattern_matrix[i]
                cur_pattern_prob = 1
                for j in range(pattern_len):
                    if cur_pattern[j] != 0:
                        cur_pattern_prob *= prob_input_pattern_vector[j]

                cur_pattern_prob *= self.gamma
                probability[temp_group] = max(cur_pattern_prob, probability[temp_group])

        # calculate resulting output
        summ_prob = sum(probability)
        for i in range(len(probability)):
            probability[i] /= summ_prob

        self.set_output_data(probability)
