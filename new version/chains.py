import logging
log = logging.getLogger(__name__)


class MarkovNode:
    def __init__(self, index):
        self.index = index
        self.parent = None
        self.strongest_connect = 0
        self.chain = None
        self.children = []

    def add_parent(self, parent):
        self.parent = parent
        parent.children.append(self)

    def __str__(self):
        return "{}  p:{}-{}\n{}".format(self.index, self.parent.index if self.parent is not None else None, self.strongest_connect,
                                          [child.index for child in self.children])


class MarkovChains:
    def __init__(self, max_num):
        self.num = 0
        self.max_num = max_num
        self.nodes = []
        self.chains = [[] for _ in range(max_num)]

    def __str__(self):
        string = "{}\n{}".format(self.num, self.chains)
        for node in self.nodes:
            string += "\n----------------------------------\n" + str(node)
        return string + "\n----------------------------------"

    def add_node(self, prev_index, cur_index):
        log.debug("adding node {}->{}".format(prev_index, cur_index))
        new = MarkovNode(cur_index)
        self.nodes.append(new)
        for (i, chain) in enumerate(self.chains):
            if not chain:
                chain.append(new.index)
                new.chain = i
                return
        parent = self.nodes[prev_index]
        new.add_parent(parent)
        new.chain = parent.chain
        self.chains[new.chain].append(new.index)

    def strengthen_connect(self, prev_index, cur_index, val):
        log.debug("strengthen_connect {}->{} {}".format(prev_index, cur_index, val))
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

        log.debug(self.__str__())

    def reconnect(self, node, new_parent, new_connect):
        log.debug("reconnect {}->{} {}".format(node.index, new_parent.index, new_connect))
        node.strongest_connect = new_connect
        if node.parent != new_parent:  # is not?
            if node.parent == node:
                node.parent = node
            else:
                if node.parent is not None:
                    node.parent.children.remove(node)
                node.add_parent(new_parent)
        log.debug(self.__str__())

    def move_root(self, node, dest_chain):
        self.move(node, dest_chain, node)

    # move node and its children to another chain

    def move(self, node, dest_chain, root=None):
        log.debug("move {}:{}->{}".format(node.index, node.chain, dest_chain))
        self.chains[node.chain].remove(node.index)
        node.chain = dest_chain
        self.chains[node.chain].append(node.index)
        for child in node.children:
            if child != root:
                self.move(child, dest_chain, root)
        log.debug(self.__str__())
