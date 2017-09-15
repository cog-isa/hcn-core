import classNode
import classLayer

class TestingStructure:
    def __init__(self, num_layers, data_matrix):
        self.__num_layers = num_layers
        self.__layers = [None] * num_layers
        self.__layer_output = None

        self.__study_layer = 1
        self.__study_end = False

        num_groups = [1, 8, 4]

        for i in range(num_layers):
            height, width,  len_h, len_w = data_matrix[i]
            prev_layer = self.__layers[i - 1]
            self.__layers[i] = classLayer.Layer(height, width, prev_layer,
                                     len_h, len_w, num_temp_groups = num_groups[i])

    def image_request(self, image):
        classLayer.initialize_input_layer(self.__layers[0], image)
        if (not self.__study_end):
            self._study_request()
            print(self.__study_layer, "study in process")
            return

        for i in range(1, self.__num_layers):
            self.__layers[i].request()

        output_node = self.__layers[-1].node_matrix[0][0]

        print("output ready")
        print(output_node.get_output())

        norm_output = output_node.get_output()
        max_val = max(norm_output)
        norm_output = [1 * (norm_output[i] == max_val and max_val != 0)
                       for i in range(len(norm_output))]
        print(norm_output)
        return

    def _study_request(self):
        for i in range(1, self.__study_layer + 1):
            self.__layers[i].request()

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

    model = TestingStructure(number_layers, data_matrix)
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