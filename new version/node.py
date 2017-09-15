import numpy as np
from functools import reduce
from bitarray import bitarray
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

#forget old connections
#bias of connection
from data_loader import Loader

class HTMNetwork:
    def __init__(self, loader, weights=None, shape=None, shape_max_chains=None):
        if shape is None:
            shape = [16*16, 16*4, 16]
        if weights is None:
            weights = [np.zeros(z) for z in zip(shape[:-1], shape[1:])]
        if shape_max_chains is None:
            shape_max_chains = [5, 5, 5]
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
            x = np.array([[220, 200, 0], [170, 155, 0], [200, 170, 100]])/255
            node = self.network[0][0]
            node.process_forward(x)
            node.process_forward(x)
            node.process_forward(np.fliplr(x))
            node.process_forward(x)
            node.process_forward(np.fliplr(x))
            node.process_forward(x)


#         for node in self.network[0]:
#               node


class Node:
    def __init__(self, max_num_chains=5):
        self.max_num_chains = max_num_chains
        self.markov_graph = np.empty((0, 0))
        self.normalized_markov_graph = np.empty((0, 0))
        # mb 3d-matrix is better
        self.patterns = []
        # value
        self.proc_alpha = 1
        self.prev_pattern_index = None
        self.node_input = np.empty((0, 0))
        self.alpha = np.empty(0)
        self.pattern_likelihood = np.empty(0)
        self.markov_chains = MarkovChains(max_num_chains)

# empty pattern is another pattern
    def process_forward(self, node_input, learn_mode=True):
        if not self.patterns and learn_mode:
            self.add_pattern(self.input_to_pattern(node_input))
            return 0
        else:
            self.calc_pattern_likelihood(node_input)
            ff_likelihood = self.calc_feedforward_likelihood()
            if learn_mode:
                self.add_pattern(self.input_to_pattern(node_input))
            self.normalize_markov_graph()
            log.debug("alpha: \n {}".format(self.alpha))
            log.debug("n_m_graph: \n {}".format(self.normalized_markov_graph))
            self.proc_alpha = self.alpha.dot(self.normalized_markov_graph)
            log.debug("proc_alpha: \n {}".format(self.proc_alpha))

        return ff_likelihood

    def input_to_pattern(self, node_input: np.array) -> np.array:

        self.node_input = node_input
        chosen_pos = np.argmax(node_input, 1)  # array of maximum's indexes

        pattern = np.zeros(node_input.shape, bool)
        for x in enumerate(chosen_pos):
            pattern[x] = (node_input[x] != 0)
        # its possible use array-indexes instead loop
        # pattern[range(len(pattern)), pattern] = (node_input[pattern] != 0)

        return pattern

    def normalize_markov_graph(self):
        norm = self.markov_graph.sum(1)
        # no division by zero
        norm[norm == 0] = 1
        log.debug("norm: {}".format(norm))
        self.normalized_markov_graph = self.markov_graph / norm[:, None]

    def calc_pattern_likelihood(self, node_input):
        #unpythonic
        self.pattern_likelihood = np.array([reduce(lambda x, y: x*y,
                                             node_input[np.array(pattern)]) for pattern in self.patterns])
        log.debug("likelihood:\n {}".format(self.pattern_likelihood))
    def calc_feedforward_likelihood(self):
        ff_likelihood = np.zeros(self.max_num_chains)

        self.alpha = np.multiply(self.proc_alpha, self.pattern_likelihood)
        log.debug("alpha#: {}".format(self.alpha))
        # numpy?
        for (pattern, chain) in enumerate(self.markov_chains.chain_nums):
            ff_likelihood[chain] += self.alpha[pattern]

        return ff_likelihood

    def add_pattern(self, pattern):
        # analyze exception?
        # pattern clustering
        log.debug("Pattern: \n {}".format(pattern))
        prev_index = self.prev_pattern_index
        pattern = pattern.tolist()

        try:
            index = self.patterns.index(pattern)
        except ValueError:
            log.info("adding new pattern")
            index = len(self.markov_graph)
            self.markov_graph = np.pad(self.markov_graph, ((0, 1), (0, 1)), 'constant')
            self.alpha = np.append(self.alpha, 1)
            log.debug("alpha!: \n {}".format(self.alpha))
            self.patterns.append(pattern)
            self.markov_chains.add_node(prev_index, index)
        else:
            self.markov_chains.strengthen_connect(prev_index, index)

        if prev_index != None :
            self.markov_graph[prev_index, index] += 1

        self.prev_pattern_index = index

# online clustering


class MarkovNode:
    def __init__(self, index, parent_index):
        self.index = index
        self.strongest_connect = 0
        self.parent = parent_index
        self.children = []

class MarkovChains:
    def __init__(self, max_num):
        self.num = 0
        self.max_num = max_num
        self.nodes = []
        self.chain_nums = []
        self.empty_chain = []

    def add_node(self, prev_index, cur_index):
        self.nodes.append(MarkovNode(prev_index, cur_index))
        if self.num < self.max_num:
            self.chain_nums.append(self.num if not self.empty_chain else self.chain_nums[self.empty_chain.pop()])
        else:
            self.chain_nums.append(self.chain_nums[prev_index])

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
            self.reconnect(prev_node, cur_index)
            if self.chain_nums[prev_index] != self.chain_nums[cur_index]:
                # older chain is more likely to absorb younger one
                if prev_node.strongest_connect < cur_node.strongest_connect:
                    self.move(prev_node, self.chain_nums[cur_index])
                else:
                    self.move(cur_node, self.chain_nums[prev_index])

        if new_connect > cur_node.strongest_connect:
            self.reconnect(cur_node, prev_index)

    def reconnect(self, node, new_parent):
        # new connect can be only 1 stronger, I suppose
        node.strongest_connect += 1
        # do nothing in case of same parent
        if node.parent != new_parent:
            if node.parent is not None:
                node.parent.children.remove(node.index)
            node.parent = new_parent
            node.parent.children.add(node.index)

    # move node and its children to another chain
    def move(self, node, dest_chain):
        old_chain = self.chain_nums[node.index]
        self.chain_nums[node.index] = dest_chain
        for index in node.children:
            self.chain_nums[index] = dest_chain
        if old_chain not in self.chain_nums:
            self.empty_chain.append(old_chain)

if __name__ == "__main__":
    np.set_printoptions(threshold=2000, linewidth=300, precision=3)
    l = Loader()
    n = HTMNetwork(l)
    n.generate_data()
    n.start()