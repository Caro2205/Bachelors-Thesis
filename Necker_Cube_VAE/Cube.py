'''
Author: Tim Gerne
gernetim@gmail.com
'''


import numpy as np
import pygame
import time
import random

'''
Used to create different Datasets of Cubes used for the VAE
'''

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_GREEN = (161, 181, 107)
COL_CLOSE = RED
COL_FAR = LIGHT_GREEN
WIDTH = 500
HEIGHT = 500
CORNER_POS = [WIDTH / 2, HEIGHT / 2]  # used to have cube drawn in the middle of the screen
FAC_RAD_TO_DEG = np.pi / 180

################# Definiton of helping vectors that are used to define the corneres of a cube object ###################
base_vectors = np.zeros((0, 3))

# for z in (-1, 1):
#     for x in (-1, 1):
#         for y in (-1, 1):
#             base_vectors = np.vstack((base_vectors, (x, y, z)))

base_vectors_lst = [(-1, -1, -1),
                    ( 1, -1, -1),
                    ( 1,  1, -1),
                    (-1,  1, -1),
                    (-1,  1,  1),
                    ( 1,  1,  1),
                    ( 1, -1,  1),
                    (-1, -1,  1)]

for vec in base_vectors_lst:
    base_vectors = np.vstack((base_vectors, vec))

##################################################################


class Cube:
    # defines Cube object by its 8 corners
    # based on https://math.stackexchange.com/questions/107778/simplest-equation-for-drawing-a-cube-based-on-its-center-and-or-other-vertices
    def __init__(self, center, side_length, visibility=[1, 1, 1, 1, 1, 1, 1, 1]):
        self.sidelength = side_length
        self.corners = np.zeros((0, 4))
        self.coords = np.zeros((0, 3))
        self.vis = np.zeros((0, 1))

        for i in range(0, 8):
            vis_array = np.array([visibility[i]])
            self.vis = np.vstack((self.vis, vis_array))

        for vector in base_vectors:
            new_coord = center + (side_length / 2) * vector
            self.coords = np.vstack((self.coords, new_coord))

        self.corners = np.hstack((self.coords, self.vis))

    def delete_corners(self):
        for row in self.corners:
            if row[3] == 0:
                row[0: 3] = 0

    def print_corners(self):
        for corner in self.corners:
            print(corner)

    def print_coords(self):
        for coord in self.coords:
            print(coord)

    def rotate_x(self, theta):
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, np.cos(theta * FAC_RAD_TO_DEG), np.sin(theta * FAC_RAD_TO_DEG)],
                                      [0, -np.sin(theta * FAC_RAD_TO_DEG), np.cos(theta * FAC_RAD_TO_DEG)]])
        new_coords = np.zeros((0, 3))
        for coord in self.coords:
            coord_transp = np.array([[x] for x in coord])
            new_coords_transp = np.matmul(rotation_matrix_x, coord_transp)
            new_coord = new_coords_transp.T
            new_coords = np.vstack((new_coords, new_coord))
        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))

    def rotate_y(self, theta):
        rotation_matrix_y = np.array([[np.cos(theta * FAC_RAD_TO_DEG), 0, -np.sin(theta * FAC_RAD_TO_DEG)],
                                      [0, 1, 0],
                                      [np.sin(theta * FAC_RAD_TO_DEG), 0, np.cos(theta * FAC_RAD_TO_DEG)]])
        new_coords = np.zeros((0, 3))
        for coord in self.coords:
            coord_transp = np.array([[x] for x in coord])
            new_coord_transp = np.matmul(rotation_matrix_y, coord_transp)
            new_coord = new_coord_transp.T
            new_coords = np.vstack((new_coords, new_coord))
        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))

    def rotate_z(self, theta):
        rotation_matrix_z = np.array([[np.cos(theta * FAC_RAD_TO_DEG), -np.sin(theta * FAC_RAD_TO_DEG), 0],
                                      [np.sin(theta * FAC_RAD_TO_DEG), np.cos(theta * FAC_RAD_TO_DEG), 0],
                                      [0, 0, 1]])
        new_coords = np.zeros((0, 3))
        for coord in self.coords:
            coord_transp = np.array([[x] for x in coord])
            new_coord_transp = np.matmul(rotation_matrix_z, coord_transp)
            new_coord = new_coord_transp.T
            new_coords = np.vstack((new_coords, new_coord))
        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))

    def print_cube(self, scale=1, title="cube"):
        corner_size = 4
        pygame.init()

        screen = pygame.display.set_mode([WIDTH, HEIGHT])
        pygame.display.set_caption(title)
        screen.fill(WHITE)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            screen.fill(WHITE)
            # front and back
            # for i in (0, 1):
            #     pygame.draw.polygon(screen, RED,
            #                         ((self.corners[i][0] * scale + CORNER_POS[0], self.corners[i][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+4][0] * scale + CORNER_POS[0], self.corners[i+4][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+6][0] * scale + CORNER_POS[0], self.corners[i+6][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+2][0] * scale + CORNER_POS[0], self.corners[i+2][1] * scale + CORNER_POS[1])))
            #
            # # right and left
            # for i in (0, 4):
            #     pygame.draw.polygon(screen, RED,
            #                         ((self.corners[i][0] * scale + CORNER_POS[0], self.corners[i][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+1][0] * scale + CORNER_POS[0], self.corners[i+1][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+3][0] * scale + CORNER_POS[0], self.corners[i+3][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+2][0] * scale + CORNER_POS[0], self.corners[i+2][1] * scale + CORNER_POS[1])))
            #
            # # top and bottom
            # for i in (0, 2):
            #     pygame.draw.polygon(screen, RED,
            #                         ((self.corners[i][0] * scale + CORNER_POS[0], self.corners[i][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+1][0] * scale + CORNER_POS[0], self.corners[i+1][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+5][0] * scale + CORNER_POS[0], self.corners[i+5][1] * scale + CORNER_POS[1]),
            #                          (self.corners[i+4][0] * scale + CORNER_POS[0], self.corners[i+4][1] * scale + CORNER_POS[1])))
            #
            # parallel to x-axis
            for i in (0, 2, 4, 6):
                adj_corner = 1
                col = COL_CLOSE if self.corners[i][2] > 0 else COL_FAR
                if self.corners[i][3] == 1 and self.corners[i + adj_corner][3] == 1:
                    pygame.draw.line(screen, col, (self.corners[i][0] * scale + CORNER_POS[0],
                                                   self.corners[i][1] * scale + CORNER_POS[1]),
                                     (self.corners[i + adj_corner][0] * scale + CORNER_POS[0],
                                      self.corners[i + adj_corner][1] * scale + CORNER_POS[1]))

            # parallel to y-axis
            for i in (1, 5):
                adj_corner = 1
                col = COL_CLOSE if self.corners[i][2] > 0 else COL_FAR
                if self.corners[i][3] == 1 and self.corners[i + adj_corner][3] == 1:
                    pygame.draw.line(screen, col, (self.corners[i][0] * scale + CORNER_POS[0],
                                                   self.corners[i][1] * scale + CORNER_POS[1]),
                                     (self.corners[i + adj_corner][0] * scale + CORNER_POS[0],
                                      self.corners[i + adj_corner][1] * scale + CORNER_POS[1]))
            for i in (0, 4):
                adj_corner = 3
                col = COL_CLOSE if self.corners[i][2] > 0 else COL_FAR
                if self.corners[i][3] == 1 and self.corners[i + adj_corner][3] == 1:
                    pygame.draw.line(screen, col, (self.corners[i][0] * scale + CORNER_POS[0],
                                                   self.corners[i][1] * scale + CORNER_POS[1]),
                                     (self.corners[i + adj_corner][0] * scale + CORNER_POS[0],
                                      self.corners[i + adj_corner][1] * scale + CORNER_POS[1]))

            # parallel to z-axis
            for i in (0, 1, 2, 3):
                adj_corner = 7
                #col = COL_CLOSE if self.corners[i][2] > 0 else COL_FAR
                if self.corners[i][3] == 1 and self.corners[adj_corner - i][3] == 1:
                    pygame.draw.line(screen, BLACK, (self.corners[i][0] * scale + CORNER_POS[0],
                                                     self.corners[i][1] * scale + CORNER_POS[1]),
                                     (self.corners[adj_corner - i][0] * scale + CORNER_POS[0],
                                      self.corners[adj_corner - i][1] * scale + CORNER_POS[1]))

            for corner in self.corners:
                col = COL_CLOSE if corner[2] > 0 else COL_FAR
                if corner[3] == 1:
                    pygame.draw.circle(screen, col, (corner[0] * scale + CORNER_POS[0],
                                                     corner[1] * scale + CORNER_POS[1]), corner_size, 0)

            pygame.display.update()

        pygame.image.save(screen, 'cube_version_1.png')
        pygame.quit()

    def add_noise(self, intensity):
        new_coords = np.zeros((0, 3))
        for [x, y, z, vis] in self.corners:
            noise_x = np.random.normal(0, 1) * (intensity * self.sidelength)
            noise_y = np.random.normal(0, 1) * (intensity * self.sidelength)
            noise_z = np.random.normal(0, 1) * (intensity * self.sidelength)

            x += noise_x
            y += noise_y
            z += noise_z

            new_coord = np.hstack((x, y, z))
            new_coords = np.vstack((new_coords, new_coord))

        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))


    def delete_all_z(self):
        new_coords = np.zeros((0, 3))
        for [x, y, z, vis] in self.corners:
            new_coord = np.stack((x, y, 0))
            new_coords = np.vstack((new_coords, new_coord))

        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))




def main():
    # define multiple cubes with sidelength between 0.5 and 3
    lengths = []
    n_cubes = 100
    center_0 = (0, 0, 0)
    data_cubes = []  # is a list with all the cube objects for the dataset
    target_cubes = [] # identical to datacubes but with all corners and without noise
    noise_intens = 0.8#0.2#0.4 #0.1


#    range of sidelength of cubes in training data
    # for i in np.linspace(0.5, 3, n_cubes):
    #     lengths.append(i)
    #
    #
    # # create training data cubes
    # visibility = [1, 1, 1, 1, 1, 1, 1, 1]
    # for i in range(0, n_cubes, 1):          # Variante 1 alle Ecken
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(20)
    #     cube.rotate_y(30)
    #     target_cubes.append(cube)
    #
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(20)
    #     cube.rotate_y(30)
    #     cube.add_noise(intensity=noise_intens)
    #     data_cubes.append(cube)
    #
    # for i in range(0, n_cubes, 1):          # Variante 2 alle Ecken
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(-20)
    #     cube.rotate_y(-30)
    #     target_cubes.append(cube)
    #
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(-20)
    #     cube.rotate_y(-30)
    #     cube.add_noise(intensity=noise_intens)
    #     data_cubes.append(cube)
    #
    # for i in range(0, n_cubes, 1):          # Variante 1 eine zufällige Ecke fehlt
    #     visibility = [1, 1, 1, 1, 1, 1, 1, 1]
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(20)
    #     cube.rotate_y(30)
    #     target_cubes.append(cube)
    #
    #     i_del = random.randint(0, 7)
    #     visibility[i_del] = 0
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(20)
    #     cube.rotate_y(30)
    #     cube.add_noise(intensity=noise_intens)
    #     cube.delete_corners()
    #     data_cubes.append(cube)
    #
    #
    # for i in range(0, n_cubes, 1):          # Variante 2 eine zufällige Ecke fehlt
    #     visibility = [1, 1, 1, 1, 1, 1, 1, 1]
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(-20)
    #     cube.rotate_y(-30)
    #     target_cubes.append(cube)
    #
    #     i_del = random.randint(0, 7)
    #     visibility[i_del] = 0
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(-20)
    #     cube.rotate_y(-30)
    #     cube.add_noise(intensity=noise_intens)
    #     cube.delete_corners()
    #     data_cubes.append(cube)
    #
    # for i in range(0, n_cubes, 1):          # Variante 1 zwei zufällige Ecke fehlen
    #     visibility = [1, 1, 1, 1, 1, 1, 1, 1]
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(20)
    #     cube.rotate_y(30)
    #     target_cubes.append(cube)
    #
    #     i_del = list(range(0, 8))
    #     random.shuffle(i_del)
    #     i1_del = i_del[0]
    #     i2_del = i_del[1]
    #     visibility[i1_del] = 0
    #     visibility[i2_del] = 0
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(20)
    #     cube.rotate_y(30)
    #     cube.add_noise(intensity=noise_intens)
    #     cube.delete_corners()
    #     data_cubes.append(cube)
    #
    # for i in range(0, n_cubes, 1):          # Variante 2 zwei zufällige Ecke fehlen
    #     visibility = [1, 1, 1, 1, 1, 1, 1, 1]
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(-20)
    #     cube.rotate_y(-30)
    #     target_cubes.append(cube)
    #
    #     i_del = list(range(0, 8))
    #     random.shuffle(i_del)
    #     i1_del = i_del[0]
    #     i2_del = i_del[1]
    #     visibility[i1_del] = 0
    #     visibility[i2_del] = 0
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(-20)
    #     cube.rotate_y(-30)
    #     cube.add_noise(intensity=noise_intens)
    #     cube.delete_corners()
    #     data_cubes.append(cube)
    #
    # for i in range(0, n_cubes, 1):       # Variante 1 ohne z-Werte
    #     visibility = [1, 1, 1, 1, 1, 1, 1, 1]
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(20)
    #     cube.rotate_y(30)
    #     target_cubes.append(cube)
    #
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(20)
    #     cube.rotate_y(30)
    #     cube.add_noise(intensity=noise_intens)
    #     cube.delete_all_z()
    #     data_cubes.append(cube)
    #
    # for i in range(0, n_cubes, 1):  # Variante 2 ohne z-Werte
    #     visibility = [1, 1, 1, 1, 1, 1, 1, 1]
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(-20)
    #     cube.rotate_y(-30)
    #     target_cubes.append(cube)
    #
    #     cube = Cube(center_0, lengths[i], visibility)
    #     cube.rotate_x(-20)
    #     cube.rotate_y(-30)
    #     cube.add_noise(intensity=noise_intens)
    #     cube.delete_all_z()
    #     data_cubes.append(cube)


# # TESTDATASET
# # '''
#     vis_list = []
#     vis_list.append([1, 1, 1, 1, 1, 1, 1, 1])
#     vis_list.append([0, 1, 1, 1, 1, 1, 1, 1])
#     vis_list.append([1, 1, 0, 1, 1, 1, 1, 1])
#     vis_list.append([0, 1, 0, 1, 1, 1, 1, 1])
#     vis_list.append([0, 0, 0, 1, 1, 1, 1, 1])
#     vis_list.append([0, 0, 0, 0, 1, 1, 1, 1])
#     vis_list.append([0, 0, 0, 0, 0, 1, 1, 1])
#     vis_list.append([0, 0, 0, 0, 0, 0, 1, 1])
#     vis_list.append([0, 0, 0, 0, 0, 0, 0, 1])
#     vis_list.append([1, 1, 1, 1, 1, 1, 1, 0])
#     vis_list.append([1, 1, 1, 1, 1, 1, 0, 0])
#     vis_list.append([1, 1, 1, 1, 1, 0, 0, 0])
#     vis_list.append([1, 1, 1, 1, 0, 0, 0, 0])
#     vis_list.append([1, 0, 1, 0, 1, 0, 1, 0])
#     vis_list.append([0, 1, 0, 1, 0, 1, 0, 1])
#     vis_list.append([1, 1, 0, 0, 0, 1, 1, 1])
#     vis_list.append([1, 0, 1, 0, 0, 1, 0, 1])
#
#
#
#
#     data_cubes = []
#     target_cubes =[]
#
#     for i in range(len(vis_list)):
#         cube_1 = Cube((0, 0, 0), 1, vis_list[i])
#         cube_1.rotate_x(20)
#         cube_1.rotate_y(30)
#         target_cubes.append(cube_1)
#
#         cube_1 = Cube((0, 0, 0), 1, vis_list[i])
#         cube_1.rotate_x(20)
#         cube_1.rotate_y(30)
#         cube_1.delete_corners()
#         data_cubes.append(cube_1)
#
#         cube_2 = Cube((0, 0, 0), 1, vis_list[i])
#         cube_2.rotate_x(-20)
#         cube_2.rotate_y(-30)
#         target_cubes.append(cube_2)
#
#         cube_2 = Cube((0, 0, 0), 1, vis_list[i])
#         cube_2.rotate_x(-20)
#         cube_2.rotate_y(-30)
#         cube_2.delete_corners()
#         data_cubes.append(cube_2)
#
# #   ohne z-koordianten
#     for i in range(len(vis_list)):
#         cube_1 = Cube((0, 0, 0), 1, vis_list[i])
#         cube_1.rotate_x(20)
#         cube_1.rotate_y(30)
#         target_cubes.append(cube_1)
#
#         cube_1 = Cube((0, 0, 0), 1, vis_list[i])
#         cube_1.rotate_x(20)
#         cube_1.rotate_y(30)
#         cube_1.delete_corners()
#         cube_1.delete_all_z()
#         data_cubes.append(cube_1)
#
#         cube_2 = Cube((0, 0, 0), 1, vis_list[i])
#         cube_2.rotate_x(-20)
#         cube_2.rotate_y(-30)
#         target_cubes.append(cube_2)
#
#         cube_2 = Cube((0, 0, 0), 1, vis_list[i])
#         cube_2.rotate_x(-20)
#         cube_2.rotate_y(-30)
#         cube_2.delete_corners()
#         cube_2.delete_all_z()
#         data_cubes.append(cube_2)

################### different rotations #########################
# # cube with no rotation
#     cube_1 = Cube((0, 0, 0), 1)
#     data_cubes.append(cube_1)
#     target_cubes.append(cube_1)
#
# # cubes with less rotation in direction of cube 1
#     cube_1 = Cube((0, 0, 0), 1)
#     cube_1.rotate_x(5)
#     cube_1.rotate_y(5)
#     data_cubes.append(cube_1)
#     target_cubes.append(cube_1)
#
#     cube_1 = Cube((0, 0, 0), 1)
#     cube_1.rotate_x(10)
#     cube_1.rotate_y(10)
#     data_cubes.append(cube_1)
#     target_cubes.append(cube_1)
#
#     cube_1 = Cube((0, 0, 0), 1)
#     cube_1.rotate_x(15)
#     cube_1.rotate_y(15)
#     data_cubes.append(cube_1)
#     target_cubes.append(cube_1)
#
#     cube_1 = Cube((0, 0, 0), 1)
#     cube_1.rotate_x(20)
#     cube_1.rotate_y(20)
#     data_cubes.append(cube_1)
#     target_cubes.append(cube_1)
#
# # cubes with less rotation in direction of cube 2
#     cube_2 = Cube((0, 0, 0), 1)
#     cube_2.rotate_x(-5)
#     cube_2.rotate_y(-5)
#     data_cubes.append(cube_2)
#     target_cubes.append(cube_2)
#
#     cube_2 = Cube((0, 0, 0), 1)
#     cube_2.rotate_x(-10)
#     cube_2.rotate_y(-10)
#     data_cubes.append(cube_2)
#     target_cubes.append(cube_2)
#
#     cube_2 = Cube((0, 0, 0), 1)
#     cube_2.rotate_x(-15)
#     cube_2.rotate_y(-15)
#     data_cubes.append(cube_2)
#     target_cubes.append(cube_2)
#
#     cube_2 = Cube((0, 0, 0), 1)
#     cube_2.rotate_x(-20)
#     cube_2.rotate_y(-20)
#     data_cubes.append(cube_2)
#     target_cubes.append(cube_2)



    #
# # create training data file
    curr_time = time.strftime("%Y_%m_%d-%H_%M_%S")

    filename = "data_" + curr_time + ".txt"
    path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/Feature-Binding-on-Necker-Cube/training_data/' + filename
    with open(path, 'a') as file:
        for cube in data_cubes:
            corner = []
            for [x, y, z, vis] in cube.corners:
                corner.append(round(x, 5))
                corner.append(round(y, 5))
                corner.append(round(z, 5))
                corner.append(round(vis, 5))
            print(*corner, sep=',', file=file)
    file.close()

    filename = "target_" + curr_time + ".txt"
    path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/Feature-Binding-on-Necker-Cube/training_data/' + filename
    with open(path, 'a') as file:
        for cube in target_cubes:
            corner = []
            for [x, y, z, vis] in cube.corners:
                corner.append(round(x, 5))
                corner.append(round(y, 5))
                corner.append(round(z, 5))
                corner.append(round(vis, 5))
            print(*corner, sep=',', file=file)
    file.close()

if __name__ == "__main__":
    #main()

    # cube_1 = Cube((0, 0, 0), 1, [1, 1, 1, 1, 1, 1, 1, 1])
    # cube_1.rotate_x(20)
    # cube_1.rotate_y(30)
    # cube_1.print_cube(scale=300)

    cube_2 = Cube((0, 0, 0), 1, [1, 1, 1, 1, 1, 1, 1, 1])
    cube_2.rotate_x(-20)
    cube_2.rotate_y(-30)
    cube_2.print_cube(scale=300)
