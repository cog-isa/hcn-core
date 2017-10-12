import numpy as np
from sklearn.cluster import MiniBatchKMeans
from functools import reduce
from bitarray import bitarray
import logging
from chains import *

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
# log.disabled = True

# forget old connections
# bias to cut connection
# bias to start connection
from data_loader import Loader


class HTMNetwork:
    def __init__(self, loader, weights=None, shape=None, shape_max_chains=None, shape_max_patterns=None):
        if shape is None:
            shape = [16 * 16, 16 * 4, 16]
        if weights is None:
            weights = [np.zeros(z) for z in zip(shape[:-1], shape[1:])]
        if shape_max_chains is None:
            shape_max_chains = [3, 3, 3]
        if shape_max_patterns is None:
            shape_max_patterns = [8, 8, 8]
        self.shape = shape
        self.weights = weights
        self.shape_max_chains = shape_max_chains
        self.loader = loader
        self.labels = []
        self.train = []
        self.movie = None

        self.network = [[Node() for _ in range(size)] for size in self.shape]
        self.chains = [MarkovChains(max) for max in shape_max_chains]

    def generate_data(self, n=1):
        self.train, self.labels = self.loader.load_train()
        self.movie = self.loader.simple_movie(self.train[:n])

    def start(self, n=1000):
        # for frame in self.movie:
        # numbers
        # input = np.split(np.concatenate(np.array_split(frame, 16), 1), 16*16, 1)
        vert = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        hor = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        diag = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        an_ver1 = self.loader.animate(vert)[0][:-1]
        an_ver2 = self.loader.animate(vert)[1][:-1]
        an_hor1 = self.loader.animate(hor)[0][:-1]
        an_hor2 = self.loader.animate(hor)[1][:-1]
        an_diag1 = self.loader.animate(diag)[0][:-1]
        an_diag2 = self.loader.animate(diag)[1][:-1]
        examples = [an_ver1, an_ver2, an_hor1, an_hor2, an_diag1, an_diag2]
        choice = np.random.randint(6, size=100)
        print(choice)
        input = reduce((lambda x, y: x + y), [examples[i] for i in choice])
        for i in input:
            print(i)

        star = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
                np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])]

        # input = reduce((lambda x, y: x + y), [an1 for _ in range(2)] + [an2 for _ in range(2)]
        #                + [star] + [an1] + [an2] + [star for _ in range(2)] + [an1])
        # for x in input:
        #    print(x)
        node = InputNode()
        for x in input:
            node.process_forward(x / 1.0)


# for node in self.network[0]:
#               node

class Node:
    def __init__(self, max_num_patterns=200, max_num_chains=15, input_len=9):
        self.max_num_chains = max_num_chains
        self.max_num_patterns = max_num_patterns
        self.markov_graph = np.empty((0, 0))
        self.normalized_markov_graph = np.empty((0, 0))
        # mb 3d-matrix is better
        self.patterns = []
        # value
        self.prev_pattern_index = None
        self.node_input = np.empty((0, 0))
        self.alpha = np.empty(0)
        self.pattern_likelihood = np.empty(0)
        self.markov_chains = MarkovChains(max_num_chains)

        self.clust = MiniBatchKMeans(n_clusters=self.max_num_patterns, verbose=1, batch_size=1)
        self.clust.partial_fit(np.random.rand(self.max_num_patterns, input_len))
        self.labels = []
        self.alpha_0 = 0.01
        self.min_weight = 0.01
        self.min_input = 0

    # empty pattern is another pattern
    def process_forward(self, node_input, learn_mode=True):
        for (n, label) in enumerate(self.labels):
            log.debug("#{}\n{}".format(n, self.clust.cluster_centers_[label].reshape(3, 3)))
        # log.debug("Stored patterns:\n {}".format(self.clust.cluster_centers_[self.labels, :]))

        node_input[node_input == 0] = self.min_input

        if learn_mode:
            self.add_pattern(node_input)
        self.calc_pattern_likelihood(node_input)
        ff_likelihood = self.calc_feedforward_likelihood()

        return ff_likelihood

    # now input is pattern, kinda

    # def input_to_pattern(self, node_input: np.array) -> np.array:
    #
    #     self.node_input = node_input
    #     chosen_pos = np.argmax(node_input, 1)
    #
    #     pattern = np.zeros(node_input.shape, bool)
    #     for x in enumerate(chosen_pos):
    #         pattern[x] = (node_input[x] != 0)
    # #its possible use array-indexes instead loop
    # #pattern[range(len(pattern)), pattern] = (node_input[pattern] != 0)
    #     return pattern

    def calc_pattern_likelihood(self, node_input):
        patterns = self.clust.cluster_centers_
        # self.pattern_likelihood = np.array([np.vdot(patterns[label], node_input) for label in self.labels])
        self.pattern_likelihood = np.dot(patterns[self.labels], node_input.reshape(-1, 1))
        log.debug("likelihood:\n {}".format(self.pattern_likelihood))

    def calc_feedforward_likelihood(self):
        log.info("Calculating feedforward_likelihood")
        ff_likelihood = np.zeros(self.max_num_chains)

        self.normalize_markov_graph()
        log.debug("n_m_graph: \n {}".format(self.normalized_markov_graph))

        for (n, chain) in enumerate(self.markov_chains.chains):
            if chain:
                log.debug("chain #{}: {}".format(n, chain))
                # can be written in 1 line
                proc_alpha = np.dot(self.alpha[chain], self.normalized_markov_graph[np.ix_(chain, chain)])
                log.debug("proc_alpha: {}".format(proc_alpha))
                self.alpha[chain] = np.multiply(proc_alpha, self.pattern_likelihood[chain])
                ff_likelihood[n] = np.sum(self.alpha[chain])

        log.debug("alpha: {}".format(self.alpha))
        # normalization
        if np.any(ff_likelihood):
            ff_likelihood /= np.sum(ff_likelihood)

        log.debug("ff_likelihood: {}".format(ff_likelihood))
        return ff_likelihood

    def normalize_markov_graph(self):
        # standart?
        norm = self.markov_graph.sum(1)
        # no division by zero
        norm[norm == 0] = 1
        log.debug("norm: {}".format(norm))
        self.normalized_markov_graph = self.markov_graph / norm[:, None]
        self.normalized_markov_graph[self.normalized_markov_graph == 0] = self.min_weight

    def add_pattern(self, pattern):
        log.debug("Pattern: \n {}".format(pattern))
        prev_index = self.prev_pattern_index

        # pattern flat
        label = self.clust.partial_fit(pattern.reshape(1, -1)).labels_
        try:
            cur_index = self.labels.index(label)
        except ValueError:
            log.info("adding new pattern")
            cur_index = len(self.labels)
            self.labels.append(label[0])
            self.markov_graph = np.pad(self.markov_graph, ((0, 1), (0, 1)), 'constant')
            self.alpha = np.append(self.alpha, self.alpha_0)  #
            log.debug("prev_alpha!: \n {}".format(self.alpha))
            self.markov_chains.add_node(prev_index, cur_index)
        else:
            self.markov_chains.strengthen_connect(prev_index, cur_index, self.markov_graph[prev_index, cur_index] + 1)

        if prev_index != None:
            self.markov_graph[prev_index, cur_index] += 1

        self.prev_pattern_index = cur_index


# online chain clustering


class InputNode(Node):
    def calc_pattern_likelihood(self, node_input):
        patterns = self.clust.cluster_centers_[self.labels]
        pattern_likelihood = 1 / np.linalg.norm(patterns - node_input.reshape(1, -1), 2, 1)
        log.debug(pattern_likelihood)
        self.pattern_likelihood = pattern_likelihood / np.max(pattern_likelihood)
        log.debug("likelihood:\n {}".format(self.pattern_likelihood))


if __name__ == "__main__":
    np.set_printoptions(threshold=2000, linewidth=300, precision=6, suppress=True)
    l = Loader()
    n = HTMNetwork(l)
    n.generate_data()
    n.start()
