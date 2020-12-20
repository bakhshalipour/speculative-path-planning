#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib
import argparse
import itertools
import scipy.misc

nx = 20
ny = 20

PIXEL_EXP = 100
SCALE = 2000

EXPANDED = 0
SPECULATION = 1  # RGB -> B=255
USE = 2          # RGB -> G -> 255

RED_LAYER = 0
GREEN_LAYER = 1
BLUE_LAYER = 2

RED = np.array([255,0,0],dtype=np.uint8)
GREEN = np.array([0,255,0],dtype=np.uint8)
BLUE = np.array([0,0,255],dtype=np.uint8)
YELLOW = np.array([245,254,209],dtype=np.uint8)
WHITE = np.array([255,255,255],dtype=np.uint8)
BLACK = np.array([30,30,30],dtype=np.uint8)
GRAY = np.array([150,150,150],dtype=np.uint8)

PIXEL_COLOR = {
    EXPANDED: RED,    # red  -- expansion
    SPECULATION: GREEN, # spec -- green
    USE: BLUE          # use  -- blue
}

fig = plt.figure()
ax1=fig.add_subplot(1,4,1)
ax1.axis('off')
ax2=fig.add_subplot(1,4,2)
ax2.axis('off')
ax3=fig.add_subplot(1,4,3)
ax3.axis('off')
ax4=fig.add_subplot(1,4,4)
ax4.axis('off')

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Patch(facecolor=(0,0,0), edgecolor='k', label='obstacle'),
                   Patch(facecolor=(0.5,0.5,0.5), edgecolor='k', label='clear'),
                   Patch(facecolor=(0,0,1), edgecolor='k', label='open list'),
                   Patch(facecolor=(0,1,0), edgecolor='k', label='speculation'),
                   Patch(facecolor=(0,1,1), edgecolor='k', label='speculation & open'),
                   Patch(facecolor=(1,1,1), edgecolor='k', label='speculation & closed'),
                   Patch(facecolor=(1,0,1), edgecolor='k', label='no speculation & closed')]

# Create the figure
ax4.legend(handles=legend_elements, loc='upper left')

im = None


class Solution:
    def __init__(self, arguments):
        ## PARSE MAP
        map_mat = np.loadtxt(arguments.map, delimiter=',', skiprows=9)
        map_mat = map_mat.transpose()
        map_mat -= np.min(map_mat)
        map_mat /= np.max(map_mat)
        map_mat *= 50
        height, width = map_mat.shape

        ## MAP -> RGB
        self.map_matrix = np.zeros([height, width, 3], dtype=np.uint8)
        self.map_matrix[:,:,BLUE_LAYER] = map_mat
        mask = (self.map_matrix == [0.,0.,0.]).all(axis=2)
        mask_boundary = (self.map_matrix == [0.,0.,50.]).all(axis=2)
        #apply the mask to overwrite the pixels
        self.map_matrix[mask] = GRAY
        self.map_matrix[mask_boundary] = BLACK
        PIXEL_EXP = np.max(self.map_matrix)
        PIXEL_EXP = 255

        print("\\> map size > {}".format(self.map_matrix.shape))
        # self.soln_map1 = np.zeros_like(self.map_matrix)
        # self.soln_map2 = np.zeros_like(self.map_matrix)
        self.soln_map1 = self.map_matrix.copy()
        self.soln_map2 = self.map_matrix.copy()
        self.soln_map3 = self.map_matrix.copy()

        # self.soln1 = np.loadtxt(arguments.soln1, delimiter=',', skiprows=1)
        # self.soln1[:, 0] -= self.soln1[0, 0]
        # self.t1 = self.soln1[:,0]
        # self.cs1 = np.cumsum(np.ones_like(self.t1))
        # self.num_expansions1 = self.soln1.shape[0]
        self.soln1, self.t1, self.cs1, self.num_expansions1 = self.clean_up_solution(arguments.soln1)
        self.soln2, self.t2, self.cs2, self.num_expansions2 = self.clean_up_solution(arguments.soln2)
        self.soln3, self.t3, self.cs3, self.num_expansions3 = self.clean_up_solution(arguments.soln3)

        self.counter = -1
        self.iter_list = sorted([(tq, 'a', cnt) for cnt, tq in enumerate(self.t1)] +
                                [(tq, 'b', cnt) for cnt, tq in enumerate(self.t2)] +
                                [(tq, 'c', cnt) for cnt, tq in enumerate(self.t3)])
        # if len(self.iter_list) < 1000:
        # SCALE = int(len(self.iter_list) / 100)
        print("\\> total expanded nodes > {}; {}".format(len(self.iter_list)/2,
                                                       len(self.iter_list)/2))
        # self.iter_idx = [i for i,_ in self.iter_list]
        # idx = np.diff([0] + [i for i,_,_ in self.iter_list])
        # dt1 = min(np.diff(self.t1))
        # dt2 = min(np.diff(self.t2))
        # idx[idx < min(dt1, dt2)] = 0
        # idx[idx > 0] = 1
        # idx = np.cumsum(idx)
        # self.iter_idx = idx
        # self.iter_idx_unique = np.unique(idx)
        self.start_index = 0
        self.done = False
        self.frames = int(len(self.iter_list) / SCALE)
        ## PARSE TITLES
        self.label1 = self.parse_label(arguments.soln1, self.t1[-1])
        self.label2 = self.parse_label(arguments.soln2, self.t2[-1])
        self.label3 = self.parse_label(arguments.soln3, self.t3[-1])
        ax1.title.set_text(self.label1)
        ax2.title.set_text(self.label2)
        ax3.title.set_text(self.label3)


    def parse_label(self, path, duration, debug=False):
        label = path.split('/')[-2]
        label = label.replace('t', '# threads: ').replace('-s', "\nspeculation: ")
        if debug:
            print('\\> {} > {}'.format(label, label[-1]))
        label = label[:-1] + ('OFF' if label[-1] == '0' else 'ON') + '\n ' \
            + '{0:.2f}'.format(duration) + ' sec'
        return label

    def clean_up_solution(self, s_txt):
        soln = np.loadtxt(s_txt, delimiter=',', skiprows=1)
        soln[:,1] -= soln[0,1]
        time_vec = soln[:,1]
        cs = np.cumsum(np.ones_like(time_vec))
        num_exp = time_vec.shape[0]
        return soln, time_vec, cs, num_exp


    def slide(self, j):
        self.counter = self.counter+1
        k = self.counter
        print(" >> {}/{}".format(k, int(len(self.iter_list)/SCALE)))
        current_index = k*SCALE
        start_index_ = self.start_index
        if self.done is False:
            for l in range(self.start_index, min(current_index, len(self.iter_list))):
                i, ss, c = self.iter_list[l]
                if ss == 'a':
                    t, x, y = self.soln1[c, 1], int(self.soln1[c, 2]), int(self.soln1[c, 3])
                    # if (self.soln_map1[y, x] != BLUE).all():
                    # self.soln_map1[y, x] = PIXEL_COLOR[int(self.soln1[c,0])]
                    if (PIXEL_COLOR[int(self.soln1[c,0])] == GRAY).all():
                        self.soln_map1[y, x] = WHITE
                    if (PIXEL_COLOR[int(self.soln1[c,0])] == RED).all():
                        self.soln_map1[y, x] += PIXEL_COLOR[int(self.soln1[c,0])]
                    elif (PIXEL_COLOR[int(self.soln1[c,0])] == BLUE).all():
                        self.soln_map1[y, x] = PIXEL_COLOR[int(self.soln1[c,0])]
                    else:
                        self.soln_map1[y, x] = PIXEL_COLOR[int(self.soln1[c,0])]
                if ss == 'b':
                    t, x, y = self.soln2[c, 1], int(self.soln2[c, 2]), int(self.soln2[c, 3])
                    # self.soln_map2[y, x] = PIXEL_COLOR[int(self.soln2[c,0])]
                    if (PIXEL_COLOR[int(self.soln2[c,0])] == GRAY).all():
                        self.soln_map2[y, x] = WHITE
                    if (PIXEL_COLOR[int(self.soln2[c,0])] == RED).all():
                        self.soln_map2[y, x] += PIXEL_COLOR[int(self.soln2[c,0])]
                    elif (PIXEL_COLOR[int(self.soln2[c,0])] == BLUE).all():
                        self.soln_map2[y, x] = PIXEL_COLOR[int(self.soln2[c,0])]
                    else:
                        self.soln_map2[y, x] = PIXEL_COLOR[int(self.soln2[c,0])]
                if ss == 'c':
                    t, x, y = self.soln3[c, 1], int(self.soln3[c, 2]), int(self.soln3[c, 3])
                    if (PIXEL_COLOR[int(self.soln3[c,0])] == GRAY).all():
                        self.soln_map3[y, x] = WHITE
                    if (PIXEL_COLOR[int(self.soln3[c,0])] == RED).all():
                        self.soln_map3[y, x] += PIXEL_COLOR[int(self.soln3[c,0])]
                    elif (PIXEL_COLOR[int(self.soln3[c,0])] == BLUE).all():
                        self.soln_map3[y, x] += PIXEL_COLOR[int(self.soln3[c,0])]
                    else:
                        self.soln_map3[y, x] = PIXEL_COLOR[int(self.soln3[c,0])]
                start_index_ += 1
            self.start_index = start_index_
        if current_index > len(self.iter_list):
            self.done = True
        # temp_soln1 = self.map_matrix + self.soln_map1
        # temp_soln2 = self.map_matrix + self.soln_map2
        # temp_soln1 = self.soln_map1
        # temp_soln2 = self.soln_map2
        # temp_soln3 = self.soln_map3
        # temp_soln1 = temp_soln1[:1500,:750]
        # temp_soln2 = temp_soln2[:1500,:750]
        # temp_soln3 = temp_soln3[:1500,:750]
        # return temp_soln1, temp_soln2, temp_soln3
        save = k == int(len(self.iter_list)/SCALE)
        return self.soln_map1[:1500,:750], self.soln_map2[:1500,:750], self.soln_map3[:1500,:750], save

soln = None

def init(): return animate(0)


def animate(k):
    # print(" >> k = {}".format(k))
    m1, m2, m3, save = soln.slide(k)
    im1 = ax1.imshow(m1, animated=True)
    im2 = ax2.imshow(m2, animated=True)
    im3 = ax3.imshow(m3, animated=True)
    if False:
        # print("\\> SAVING IMAGES")
        # breakpoint()
        from PIL import Image
        # Image.fromarray(m1).save('m1'+'-{:03d}'.format(k), 'PNG')
        # Image.fromarray(m2).save('m2'+'-{:03d}'.format(k), 'PNG')
        # Image.fromarray(m3).save('m3'+'-{:03d}'.format(k), 'PNG')
        # m = im1.paste(Image.fromarray(m3).paste(Image.fromarray(m3),0),0)
        # m.save('M'+'-{:03d}'.format(k), 'PNG')
    return [im1, im2, im3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str,
                        default='../path-planning/inputset/map1.txt',
                        help=' > map file')
    parser.add_argument('--soln1', type=str,
                        default='../path-planning/csv_files/t1-s0/map1.csv',
                        help=' > specify csv file "time,x,y"')
    parser.add_argument('--soln2', type=str,
                        default='../path-planning/csv_files/t16-s0/map1.csv',
                        help=' > specify csv file "time,x,y"')
    parser.add_argument('--soln3', type=str,
                        default='../path-planning/csv_files/t16-s1/map1.csv',
                        help=' > specify csv file "time,x,y"')

    arguments = parser.parse_args()
    soln = Solution(arguments)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=soln.frames, interval=20, blit=True, repeat=False)

    gif_name = arguments.map.split('/')[-1].split('.')[1] + '-' \
        + arguments.soln1.split('/')[-2] + '-' \
        + arguments.soln2.split('/')[-2] + '-' \
        + arguments.soln3.split('/')[-2]
    # anim.save(gif_name+'.gif', writer='imagemagick', fps=30)
    plt.show()
    # anim.save(gif_name+'.mp4')

    fig_ = plt.figure()
    ax_ = fig_.subplots()
    t1 = soln.t1[soln.soln1[:,0] == 0]
    t2 = soln.t2[soln.soln2[:,0] == 0]
    t3 = soln.t3[soln.soln3[:,0] == 0]
    cs1 = np.cumsum(np.ones_like(t1))
    cs2 = np.cumsum(np.ones_like(t2))
    cs3 = np.cumsum(np.ones_like(t3))
    num_expansions1 = max(cs1)
    num_expansions2 = max(cs2)
    num_expansions3 = max(cs3)
    ax_.plot(t1, cs1, '-k', label=soln.label1)
    ax_.plot(t2, cs2, '-r', label=soln.label2)
    ax_.plot(t3, cs3, '--r', label=soln.label3)
    ax_.set_xlabel('Time (sec)')
    ax_.set_ylabel('Expanded Nodes')
    ax_.set_xlim(0, max(t1[-1], t2[-1]))
    ax_.set_ylim(0, 1.1*max(num_expansions1, num_expansions2))
    ax_.legend(loc='lower right')
    # ax_.axis('equal')
    plt.show()
    # breakpoint()
