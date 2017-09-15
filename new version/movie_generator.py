"""
result file is a sequence of appears and shifts
of the input image -- movie
this programme generates 2 movies: vertical and horizontal
first line is:
dimensions and total len of the movies
total len = 2 * (h + w + 1)
_______
example
2 2 10
# input
1 0
0 1
# vertical movie _v_
0 0    0 1    1 0    0 0    0 0
0 0    0 0    0 1    1 0    0 0
# horizontal movie
0 0    0 0    1 0    0 1    0 0
0 0    1 0    0 1    0 0    0 0
_______
"""


def print_fout(fout, h, w, data):
    for i in range(h):
        fout.write(' '.join(map(str, data[i])) + "\n")
    fout.write("\n")


def generate_movie(input_name):
    fin = open("./images/" + input_name, "r")
    fout = open("./images/" + "movie_" + input_name, "w")

    height, width = map(int, fin.readline().split())
    data = [list(map(int, fin.readline().split())) for i in range(height)]
    empty_arr = [[0] * width for i in range(height)]

    print(height, width, 2 * (height + width) + 1, file = fout)

    working_arr_vert = empty_arr + data + empty_arr
    for h in range(2 * height, -1, -1):
        shift = working_arr_vert[h:h + height]
        print_fout(fout, height, width, shift)

    working_arr_hor = [empty_arr[i] + data[i] + empty_arr[i] for i in range(height)]
    for w in range(2 * width, -1, -1):
        shift = [working_arr_hor[i][w:w + width] for i in range(height)]
        print_fout(fout, height, width, shift)

    fin.close()
    fout.close()

# main
if __name__ == "__main__":
#"""
    while True:
        print("Ready! Input next name")
        name = str(input())
        generate_movie(name)
#"""