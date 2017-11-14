import logging
from collections import deque

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
        return "{} c:{} p:{}-({})\n{}".format(self.index, self.chain,
                                              self.parent.index if self.parent is not None else None,
                                              self.strongest_connect,
                                              [child.index for child in self.children])


class MarkovChains:
    def __init__(self, max_num):
        self.inert = 3
        self.max_num = max_num
        self.nodes = []
        self.chains = [[] for _ in range(max_num)]

    def __str__(self):
        string = str(self.chains)
        for node in self.nodes:
            string += "\n----------------------------------\n" + str(node)
        return string + "\n----------------------------------\n"

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
        new.strongest_connect += 1
        new.chain = parent.chain
        self.chains[new.chain].append(new.index)

    def strengthen_connect(self, prev_index, cur_index, val):
        log.debug("strengthen_connect {}->{} {}".format(prev_index, cur_index, val))
        new_connect = val
        cur_node = self.nodes[cur_index]
        prev_node = self.nodes[prev_index]

        # probably unpythonic as fuck
        # old connections are more likely to stay
        if new_connect > prev_node.strongest_connect + self.inert or \
                        new_connect > cur_node.strongest_connect + self.inert:
            if prev_node.chain != cur_node.chain:
                # older chain is more likely to absorb younger one
                if prev_node.strongest_connect < cur_node.strongest_connect:
                    self.move_root(prev_node, cur_node.chain)
                else:
                    self.move_root(cur_node, prev_node.chain)

            # optimized, but became less readable
            # if new_connect > prev_node.strongest_connect:
            #     self.reconnect(prev_node, cur_node, new_connect)
            #
            # if new_connect > cur_node.strongest_connect:
            #     self.reconnect(cur_node, prev_node, new_connect)

            if not new_connect > prev_node.strongest_connect + self.inert:
                self.reconnect(cur_node, prev_node, new_connect)
            else:
                self.reconnect(prev_node, cur_node, new_connect)
                if new_connect > cur_node.strongest_connect + self.inert:
                    self.reconnect(cur_node, prev_node, new_connect)
        # questionable solution, used to not allow overpopulation of chains
        # else:
        #     for (i, chain) in enumerate(self.chains):
        #         if not chain:
        #             if cur_node.parent in cur_node.children:
        #                 log.debug("special move")
        #                 self.move_root(cur_node, i)
        log.debug(str(self))

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
        log.debug(str(self))

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
        log.debug(str(self))

class MarkovNodeI:
    def __init__(self, index):
        self.index = index
        self.inp = []
        self.out = []
        self.act_out = []
        self.chain = []
        self.max_connect = 0

class MarkovChainI:
    def __init__(self, node):
        self.nodes = [node]
        self.max_connect = 0
        self.adj_graph = None

class MarkovGroupsI:
    def __init__(self, max_num):
        self.inert = 1
        self.max_num = max_num
        self.nodes = []
        self.chains_connect = [[[] for _ in range(max_num)] for _ in range(max_num)]
        self.groups = []
        self.empty_chains = deque(range(max_num))  # list
        self.last_chain = None

    def __str__(self):
        string = str(self.chains)
        for node in self.nodes:
            string += "\n----------------------------------\n" + str(node)
        return string + "\n----------------------------------\n"

    def check_act(self, n):
        n.act_out = [[i for i in n.act_out if n_out[i][1] >= n.max_connect - self.inert] for n_out in n.out]


    def add_node(self, prev_index, cur_index):
        log.debug("adding node {}->{}".format(prev_index, cur_index))
        new = MarkovNodeI(cur_index)
        self.nodes.append(new)
        new.chain.append(self.empty_chains.popleft())
        self.chains[new.chain].append(new.index)
        if self.prev_index is not None:
            self.strengthen_connect(self.nodes[prev_index], new)

    def strengthen_connect(self, prev, cur):
        log.debug("strengthen_connect {}->{} {}".format(prev, cur))



        # prev_out = prev.out[self.last_chain]
        # i = next((i for i, (n, c) in enumerate(prev_out) if n is cur), -1)
        # new_connect = 1
        # if i < 0:
        #     prev_out.append([cur, 1])
        # else:
        #     prev_out[i][-1] += 1
        #     new_connect = prev_out[i][1]
        # # add forgetting
        # if new_connect > max(prev.max_connect, cur.max_connect) / 2 + self.inert:
        #     if new_connect > max(prev.max_connect):
        #         prev.max_connect = new_connect
        #         self.check_act(prev)
        #     if new_connect > max(cur.max_connect):
        #         cur.max_connect = new_connect
        #         self.check_act(cur)
        #     cur.act_out.add(i)
        #     self.add_to_chain(cur)

    def chain_breakup(self, chain):
        new_chains = []
        for root in self.chains[chain]:
            visited = set()
            q = deque(root)
            while q:
                x = q.popleft()
                if x not in visited:
                    visited.add(x)
                    q.extend(n for n,_ in x.output[chain])
                    q.extend(x.inp[chain]) # not necessary substruction
            new_chains.append(visited)

        # maybe give the biggest chain old chain_num?
        max_chain = max(new_chains, key=len)
        for new_chain in new_chains:
            chain_index = chain if (new_chain is max_chain) else self.empty_chains.popleft()
            self.chains[chain_index] = new_chain
            for n in new_chain:
                n.chain = chain_index
        for


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
        log.debug(str(self))

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
        log.debug(str(self))
