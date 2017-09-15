from math import exp
import clustering

INF = 1e10

class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        return Point(self.x + other.x, self.y + other.y)


class Node:
    def __init__(self, point_s = Point(), point_f = Point(), num_groups = 4, sigma2 = 5):
        self.__start_point = point_s    # connection
        self.__finish_point = point_f   # area
        self.__sigma2 = sigma2          # sigma^2

        self.__prev_pattern_index = None
        self.__number_patterns = 0
        self.__pattern_matrix = []

        self.__Markov_graph = []
        self.__normalized_Markov_graph = []
        self.__distance_graph = []      # inversely to normalized graph

        self.__temporal_groups = []
        self.__num_groups = num_groups

        self.__output = []              # request result

    def extract_pattern(self, prev_layer):
        pattern = []
        node_matrix = prev_layer.node_matrix
        for i in range(self.__start_point.x, self.__finish_point.x):
            for j in range(self.__start_point.y, self.__finish_point.y):
                cur_line = node_matrix[i][j].get_output()
                max_value = max(cur_line)
                if max_value > 0:
                    cur_line = [1 *(cur_line[i] == max_value) for i in range(len(cur_line))]
                pattern += cur_line[:]
                ##pattern += node_matrix[i][j].get_output()
        return pattern

    def add_pattern(self, pattern):
        if pattern not in self.__pattern_matrix:
            index = self.__number_patterns
            self.__pattern_matrix.append(pattern)
            self.__number_patterns += 1
            for i in range(self.__number_patterns - 1):
                self.__Markov_graph[i] += [0]
                self.__normalized_Markov_graph[i] += [0]
                self.__distance_graph[i] += [INF]

            self.__Markov_graph.append([0] * self.__number_patterns)
            self.__normalized_Markov_graph.append([0] * self.__number_patterns)
            self.__distance_graph.append([INF] * self.__number_patterns)
        else:
            index = self.__pattern_matrix.index(pattern)

        prev_index = self.__prev_pattern_index
        if(prev_index != None):
            self.__Markov_graph[prev_index][index] += 1

        self.__prev_pattern_index = index

    def normalize_graph(self):
        for i in range(self.__number_patterns):
            number_links = sum(self.__Markov_graph[i])
            for j in range(self.__number_patterns):
                if (number_links == 0):
                    continue
                self.__normalized_Markov_graph[i][j] = self.__Markov_graph[i][j] / number_links

    def recalc_distance(self):
        for i in range(self.__number_patterns):
            for j in range(self.__number_patterns):
                if self.__normalized_Markov_graph[i][j] == 0:
                    self.__distance_graph[i][j] = INF
                else:
                    self.__distance_graph[i][j] = 1 / self.__normalized_Markov_graph[i][j]

    def partition_temporal(self, threshold = 0.5, method = 'average'):
        if len(self.__normalized_Markov_graph) == 1:
            self.__temporal_groups = [0]
            return
        self.__temporal_groups = clustering.hierarchical(self.__distance_graph, self.__num_groups)

    def get_output(self):
        return self.__output[:]

    def set_output_data(self, output):
        self.__output = output[:]

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
        dist_square = [0] * self.__number_patterns
        probability = [0] * self.__num_groups

        for i in range(self.__number_patterns):
            value = 0
            temp_group = self.__temporal_groups[i]  # temp_group >= 1
            temp_group -= 1
            cur_pattern = self.__pattern_matrix[i]
            for j in range(pattern_len):
                value += (cur_pattern[j] - pattern[j]) ** 2

            dist_square[i] = value
            new_prob = exp(-dist_square[i] / self.__sigma2)
            probability[temp_group] = max(new_prob, probability[temp_group])

        summ_prob = sum(probability)
        for i in range(len(probability)):
            probability[i] /= summ_prob

        self.set_output_data(probability)


class Layer:
    def __init__(self, height, width, prev_layer = None, len_h = 1, len_w = 1, num_temp_groups = 1):
        self.__height = height
        self.__width = width
        self.__prev_layer = prev_layer
        self.__study_mode = True

        self.node_matrix = tuple([tuple([Node(Point(len_h * i, len_w * j),
                                  Point(len_h * (i + 1), len_w * (j + 1)),
                                              num_groups= num_temp_groups)
                             for j in range(width)]) for i in range(height)])
        # len_h * len_w = connection area

    def get_dimensions(self):
        return (self.__height, self.__width)

    def is_study_mode(self):
        return self.__study_mode

    def switch_off_study(self):
        self.__study_mode = False

    def request(self):
        for i in range(self.__height):
            for j in range(self.__width):
                self.node_matrix[i][j].calculate(self.__prev_layer, self.__study_mode)


def initialize_input_layer(layer, input_data):
    # input_data = image matrix
    height, width = layer.get_dimensions()
    for i in range(height):
        for j in range(width):
            (layer.node_matrix[i][j]).set_output_data([ input_data[i][j] ])


class SimpleStructure:
    def __init__(self, num_layers, data_matrix):
        self.__num_layers = num_layers
        self.__layers = [None] * num_layers
        self.__layer_output = None

        self.__study_layer = 1
        self.__study_iter = 0
        self.__study_end = False

        #self.__max_iter = max_iter

        num_groups = [1, 4, 4]

        for i in range(num_layers):
            height, width,  len_h, len_w = data_matrix[i]
            prev_layer = self.__layers[i - 1]
            self.__layers[i] = Layer(height, width, prev_layer,
                                     len_h, len_w, num_temp_groups = num_groups[i])

    #    self.__layer_output = Layer(1, 1, self.__layers[-1], height, width)
    #    self.__layer_output.switch_off_study()

    def image_request(self, image):
        initialize_input_layer(self.__layers[0], image)
        if (not self.__study_end):
            self._study_request()
            print(self.__study_layer, self.__study_iter, "study in process")
            return

        for i in range(1, self.__num_layers):
            self.__layers[i].request()
        #self.__layer_output.request()
        output_node = self.__layers[-1].node_matrix[0][0]

        print("output ready")
        print(output_node.get_output())

        norm_output = output_node.get_output()
        max_val = max(norm_output)
        norm_output = [1 * (norm_output[i] == max_val) for i in range(len(norm_output))]
        print(norm_output)

        return

    def _study_request(self):
        ##cur_layer = self.__layers[self.__study_layer]
        ##self.__study_iter += 1
        for i in range(1, self.__study_layer + 1):
            self.__layers[i].request()
        """
        if self.__study_iter == self.__max_iter:
            cur_layer.switch_off_study()
            self.__study_layer += 1
            self.__study_iter = 0
        """

    def switch_up_study_level(self):
        cur_layer = self.__layers[self.__study_layer]
        cur_layer.switch_off_study()
        self.__study_layer += 1
        if self.__study_layer == self.__num_layers:
            self.__study_end = True
            print("ready to request")


def new_test():
    # layer0 = input_layer
    number_layers = 3
    # matrix params: height, width, len_h, len_w
    data_matrix = [[8, 8, 1, 1],
                   [2, 2, 4, 4],
                   [1, 1, 2, 2]]
    fin1_files = open("./images/file_names_test1.txt", "r")    # files = movie series
    fin2_files = open("./images/file_names_test2.txt", "r")

    num_files_test = [int(fin1_files.readline()), int(fin2_files.readline())]

    data_test = [[fin1_files.readline().rstrip() for i in range(num_files_test[0])],
                 [fin2_files.readline().rstrip() for i in range(num_files_test[1])]]

    model = SimpleStructure(number_layers, data_matrix)
    h, w, movie_len = (0, 0, 0)

    for i in range(number_layers - 1):
        for j in range(num_files_test[i]):
            fin = open("./images/" + data_test[i][j], "r")
            h, w, movie_len = map(int, fin.readline().split())
            for k in range(movie_len):
                image = [list(map(int, fin.readline().split())) for it in range(h)]
                fin.readline()
                model.image_request(image)
        model.switch_up_study_level()

    while True:
        image = [list(map(int, input().split())) for k in range(h)]
        model.image_request(image)


new_test()
"""
def test():
    # layer0 = input_layer
    number_layers = 3
    # matrix params: height, width, len_h, len_w
    data_matrix = [[4, 8, 1, 1],
                   [1, 2, 4, 4],
                   [1, 1, 1, 2]]
    movie_len = 7
    number_images = 4  # I, L, O, U
    image_sizes = (4, 8)
    max_iter = movie_len * number_images

    data_files = ["test_L.txt", "test_U.txt", "test_I.txt", "test_O.txt"]

    model = SimpleStructure(number_layers, data_matrix, max_iter)

    #image = [[0] * image_sizes[1] for i in range(image_sizes[0])]

    for L_num in range(number_layers - 1):
        for i in range(number_images):
            fin = open(data_files[i], 'r')
            for j in range(movie_len):
                image = [list(map(int, fin.readline().split())) for k in range(image_sizes[0])]
                fin.readline()
                model.image_request(image)

    while True:
        #letter = input() # letter: ex. "O"
        #fin = open("test_" + letter + ".txt", 'r')
        image = [list(map(int, input().split())) for k in range(image_sizes[0])]
        model.image_request(image)
        fin.readline()


test()
"""