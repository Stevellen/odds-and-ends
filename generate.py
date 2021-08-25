# %%
import time
import numpy as np
from numpy import zeros, ones, fill_diagonal
from sandpile import Sandpile
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from multiprocessing import Pool
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

# %%
def x_cross(size):
    fill_diagonal(pour := zeros((size,size)), 2)
    fill_diagonal(pour2 := zeros((size,size)), 2)
    pour = pour + np.fliplr(pour2)
    pour[size//2,size//2] = 2
    return pour
    
def t_cross(size):
    pour = np.zeros((size,size))
    pour[:, size//2] = pour[0, :] = 2
    return pour

def Y(size):
    pour = x_cross(size)
    pour[size//2:,:] = 0
    pour [size//2:, size//2] = 2
    return pour

def peace(size):
    pour = x_cross(size)
    pour[:size//2,:] = 0
    pour [:, size//2] = 2
    return pour
    
# %%
def animate(template, iterations, save, name):
    box = Sandpile.zeros(template.shape)
    template = Sandpile(template, template.shape)
    box.watch_sand(template, iterations, save=save, save_to=name+'.mp4')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('size', type=int)
    parser.add_argument('iter', type=int)
    args = parser.parse_args()
    parms = [
        (Y(args.size), args.iter, True, str(f'./vids/{args.size}/Y')),
        (t_cross(args.size), args.iter, True, str(f'./vids/{args.size}/t_cross')), 
        (x_cross(args.size), args.iter, True, str(f'./vids/{args.size}/x_cross')),
        (peace(args.size), args.iter, True, str(f'./vids/{args.size}/peace'))
    ]
    with Pool(processes=4) as pool:
        pool.starmap(animate, parms)

