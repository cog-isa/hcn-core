import numpy as np
from sklearn.cluster import MiniBatchKMeans
from functools import reduce
from bitarray import bitarray
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

#forget old connections
#bias to cut connection
#bias to start connection
from data_loader import Loader

class HTMNetwork:
    def __init__(self, loader, weights=None, shape=None, shape_max_chains=None, shape_max_patterns=None):
        if shape is None:
            shape = [16*16, 16*4, 16]
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

    def generate_data(self, n = 1):
        self.train, self.labels = self.loader.load_train()
        self.movie = self.loader.simple_movie(self.train[:n])

    def start(self, n=1000):
        #for frame in self.movie:
            # numbers
            # input = np.split(np.concatenate(np.array_split(frame, 16), 1), 16*16, 1)
        xs = [np.random.rand(1, 4) for _ in range(100)]
        node = self.network[0][0]
        for x in xs:
            node.process_forward(x)
#         for node in self.network[0]:
#               node

class Node:
    def __init__(self, max_num_patterns=10, max_num_chains=5, input_len=4):
        self.max_num_chains = max_num_chains
        self.max_num_patterns = max_num_patterns
        self.markov_graph = np.empty((0, 0))
        self.normalized_markov_graph = np.empty((0, 0))
        # mb 3d-matrix is better
        self.patterns = []
        # value
        self.prev_alpha = np.empty(0)
        self.prev_pattern_index = None
        self.node_input = np.empty((0, 0))
        self.alpha = np.empty(0)
        self.pattern_likelihood = np.empty(0)
        self.markov_chains = MarkovChains(max_num_chains)

        self.clust = MiniBatchKMeans(n_clusters=self.max_num_patterns, verbose=1, batch_size=1)
        self.clust.partial_fit(np.random.rand(self.max_num_patterns, input_len))
        self.labels = []

# empty pattern is another pattern
    def process_forward(self, node_input, learn_mode=True):
        log.debug("Current centers:\n {}".format(self.clust.cluster_centers_))

        if learn_mode:
            self.add_pattern(node_input)
        self.calc_pattern_likelihood(node_input)
        ff_likelihood = self.calc_feedforward_likelihood()

        return ff_likelihood

    def input_to_pattern(self, node_input: np.array) -> np.array:
        # now input is pattern, kinda

        self.node_input = node_input
        chosen_pos = np.argmax(node_input, 1)

        pattern = np.zeros(node_input.shape, bool)
        for x in enumerate(chosen_pos):
            pattern[x] = (node_input[x] != 0)
        # its possible use array-indexes instead loop
        # pattern[range(len(pattern)), pattern] = (node_input[pattern] != 0)

        return pattern

    def normalize_markov_graph(self):
        # standart?
        norm = self.markov_graph.sum(1)
        # no division by zero
        norm[norm == 0] = 1
        log.debug("norm: {}".format(norm))
        self.normalized_markov_graph = self.markov_graph / norm[:, None]

    def calc_pattern_likelihood(self, node_input):
        patterns = self.clust.cluster_centers_
        self.pattern_likelihood = np.array([np.vdot(patterns[label], node_input) for label in self.labels])
        log.debug("likelihood:\n {}".format(self.pattern_likelihood))

    def calc_feedforward_likelihood(self):
        log.info("Calculating feedforward_likelihood")

        self.normalize_markov_graph()
        log.debug("n_m_graph: \n {}".format(self.normalized_markov_graph))

        for (n, chain) in enumerate(self.markov_chains.chains):
            if chain:
                log.debug("chain #{}: {}".format(n, chain))
                self.alpha[chain] = np.dot(self.prev_alpha[chain], self.normalized_markov_graph[np.ix_(chain)])

        log.debug("alpha: {}".format(self.alpha))
        ff_likelihood = self.alpha.dot(self.pattern_likelihood)
        # normalization
        if np.any(ff_likelihood):
            ff_likelihood /= np.sum(ff_likelihood)
        self.prev_alpha = self.alpha

        log.debug("ff_likelihood: {}".format(ff_likelihood))
        return ff_likelihood

    def add_pattern(self, pattern):
        log.debug("Pattern: \n {}".format(pattern))
        prev_index = self.prev_pattern_index
        # pattern flat

        label = self.clust.partial_fit(pattern.reshape(1, -1)).labels_
        try:
            index = self.labels.index(label)
        except ValueError:
            log.info("adding new pattern")
            index = len(self.labels)
            self.labels.append(label)
            self.markov_graph = np.pad(self.markov_graph, ((0, 1), (0, 1)), 'constant')
            self.prev_alpha = np.append(self.prev_alpha, 0)
            self.alpha = np.append(self.alpha, 0)  #
            log.debug("prev_alpha!: \n {}".format(self.prev_alpha))
            self.markov_chains.add_node(prev_index, index)
        else:
            self.markov_chains.strengthen_connect(prev_index, index)

        if prev_index != None :
            self.markov_graph[prev_index, index] += 1

        self.prev_pattern_index = index

# online chain clustering

class MarkovNode:
    def __init__(self, parent_index, index):
        self.index = index
        self.strongest_connect = 0
        self.parent = parent_index
        self.children = []
        self.chain = None

class MarkovChains:
    def __init__(self, max_num):
        self.num = 0
        self.max_num = max_num
        self.nodes = []
        self.chains = [[] for _ in range(max_num)]
        # self.chain_nums = []
        # changed list repr and therefore code become simpler

    def add_node(self, prev_index, cur_index):
        new = MarkovNode(prev_index, cur_index)
        self.nodes.append(new)
        for (i, chain) in enumerate(self.chains):
            if not chain:
                chain.append(new.index)
                new.chain = i
                return
        new.chain = self.nodes[prev_index].chain
        self.chains[new.chain].append(new.index)

    def strengthen_connect(self, prev_index, cur_index):
        # ignore new connections between until enough chains :(
        if (self.num < self.max_num):
            return

        new_connect = self.markov_graph[prev_index, cur_index]
        cur_node = self.nodes[cur_index]
        prev_node = self.nodes[prev_index]

        # probably unpythonic as fuck
        # need to be restructured

        if new_connect > prev_node.strongest_connect:
            self.reconnect(prev_node, cur_index, new_connect)
            if prev_node.chain != cur_node.chain:
                # older chain is more likely to absorb younger one
                if prev_node.strongest_connect < cur_node.strongest_connect:
                    self.move(prev_node, cur_node.chain)
                else:
                    self.move(cur_node, prev_node.chain)

        if new_connect > cur_node.strongest_connect:
            self.reconnect(cur_node, prev_index, new_connect)

    def reconnect(self, node, new_parent, new_connect):
        node.strongest_connect = new_connect
        # do nothing in case of same parent
        if node.parent != new_parent:
            if node.parent is not None or node.parent != node.index:
                node.parent.children.remove(node)
                new_parent.children.add(node)
            node.parent = new_parent

    # move node and its children to another chain
    def move(self, node, dest_chain):
        old_chain = node.chain
        node.chain = dest_chain
        for child in node.children:
            child.chain = dest_chain

if __name__ == "__main__":
    np.set_printoptions(threshold=2000, linewidth=300, precision=3)
    l = Loader()
    n = HTMNetwork(l)
    n.generate_data()
    n.start()