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
        vert = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        hor = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        diag = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        xs = self.loader.simple_movie([vert for _ in range(10)] + [hor for _ in range(10)] + [diag for _ in range(10)])
        for x in xs:
            print(x)
        node = self.network[0][0]
        for x in xs:
            node.process_forward(x)
#         for node in self.network[0]:
#               node

class Node:
    def __init__(self, max_num_patterns=300, max_num_chains=4, input_len=9):
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
        self.alpha_0 = 1
        self.min_weight = 0.01

# empty pattern is another pattern
    def process_forward(self, node_input, learn_mode=True):
        for (n, label) in enumerate(self.labels):
            log.debug("#{}\n{}".format(n, self.clust.cluster_centers_[label].reshape(3, 3)))
        # log.debug("Stored patterns:\n {}".format(self.clust.cluster_centers_[self.labels, :]))

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
        self.normalized_markov_graph[self.normalized_markov_graph == 0] = self.min_weight

    def calc_pattern_likelihood(self, node_input):
        patterns = self.clust.cluster_centers_
        self.pattern_likelihood = np.array([np.vdot(patterns[label], node_input) for label in self.labels])
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
                self.alpha[chain] = np.multiply(proc_alpha, self.pattern_likelihood[chain])
                ff_likelihood[n] = np.sum(self.alpha[chain])

        log.debug("alpha: {}".format(self.alpha))
        # normalization
        if np.any(ff_likelihood):
            ff_likelihood /= np.sum(ff_likelihood)

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
            self.labels.append(label[0])
            self.markov_graph = np.pad(self.markov_graph, ((0, 1), (0, 1)), 'constant')
            self.alpha = np.append(self.alpha, self.alpha_0)  #
            log.debug("prev_alpha!: \n {}".format(self.alpha))
            self.markov_chains.add_node(prev_index, index)
        else:
            self.markov_chains.strengthen_connect(prev_index, index, self.markov_graph[prev_index, index])

        if prev_index != None :
            self.markov_graph[prev_index, index] += 1

        self.prev_pattern_index = index

# online chain clustering

class MarkovNode:
    def __init__(self, index):
        self.index = index
        self.strongest_connect = 0
        self.parent = None
        self.children = []
        self.chain = None

    def addparent(self, parent):
        self.parent = parent
        parent.children.append(self)

class MarkovChains:
    def __init__(self, max_num):
        self.num = 0
        self.max_num = max_num
        self.nodes = []
        self.chains = [[] for _ in range(max_num)]
        # self.chain_nums = []
        # changed list repr and therefore code become simpler

    def add_node(self, prev_index, cur_index):
        new = MarkovNode(cur_index)
        self.nodes.append(new)
        for (i, chain) in enumerate(self.chains):
            if not chain:
                chain.append(new.index)
                new.chain = i
                return
        parent = self.nodes[prev_index]
        new.addparent(parent)
        new.chain = parent.chain
        self.chains[new.chain].append(new.index)

    def strengthen_connect(self, prev_index, cur_index, val):
        # ignore new connections between until enough chains :(
        #if (self.num < self.max_num):
        #   return
        new_connect = val
        cur_node = self.nodes[cur_index]
        prev_node = self.nodes[prev_index]

        # probably unpythonic as fuck
        if new_connect > prev_node.strongest_connect:
            self.reconnect(prev_node, cur_node, new_connect)
            if prev_node.chain != cur_node.chain:
                # older chain is more likely to absorb younger one
                if prev_node.strongest_connect < cur_node.strongest_connect:
                    self.move_root(prev_node, cur_node.chain)
                else:
                    self.move_root(cur_node, prev_node.chain)

        if new_connect > cur_node.strongest_connect:
            self.reconnect(cur_node, prev_node, new_connect)

    def reconnect(self, node, new_parent, new_connect):
        node.strongest_connect = new_connect
        if node.parent != new_parent:  # is not?
            if node.parent == node:
                node.parent = node
            else:
                if node.parent is not None:
                    node.parent.children.remove(node)
                node.addparent(new_parent)

    def move_root(self, node, dest_chain):
        self.move(node, dest_chain, node)
    # move node and its children to another chain

    def move(self, node, dest_chain, root=None):
        log.debug("$move")
        self.chains[node.chain].remove(node.index)
        node.chain = dest_chain
        self.chains[node.chain].append(node.index)
        #print(node)
        #print(node.children)
        for child in node.children:
            if child != root:
                self.move(child, dest_chain, root)

if __name__ == "__main__":
    np.set_printoptions(threshold=2000, linewidth=300, precision=3, suppress=True)
    l = Loader()
    n = HTMNetwork(l)
    n.generate_data()
    n.start()