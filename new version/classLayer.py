import classNode
from classNode import Point

class Layer:
    def __init__(self, height, width, prev_layer = None,
                 len_h = 1, len_w = 1, num_temp_groups = 1, max_node_elem_val = 1):
        self.__height = height
        self.__width = width
        self.__prev_layer = prev_layer
        self.__study_mode = True
        self.__node_dimension = 1
        self.__index = 0
        self.__max_node_elem_val = max_node_elem_val

        if prev_layer != None:
            self.__index = prev_layer.get_index() + 1
            self.__node_dimension = prev_layer.get_node_dimension() * len_h * len_w

        self.node_matrix = tuple([tuple(
            [classNode.ClusteredNode(point_s= Point(len_h * i, len_w * j),
                                     point_f= Point(len_h * (i + 1), len_w * (j + 1)),
                                     num_groups= num_temp_groups,
                                     layer_index= self.__index,
                                     dimension= self.__node_dimension,
                                     max_val= max_node_elem_val)
             for j in range(width)]) for i in range(height)])
        # len_h * len_w = connection area

    def get_index(self):
        return self.__index

    def get_node_dimension(self):
        return self.__node_dimension

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
