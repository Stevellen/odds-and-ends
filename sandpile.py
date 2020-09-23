# %%
import numpy as np
from numpy import zeros, ones, product, fill_diagonal
from numpy.random import randint
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import warnings
warnings.filterwarnings("ignore")


# %%
class Sandpile:
    def __init__(self, points=None, shape=(3,3), max_height=3, dtype=np.int32):
        self.shape = shape
        self.max_height = max_height
        self.dtype = dtype
        self.cmap = ListedColormap(['white', 'green', 'purple', 'yellow'])
        if points is None:
            self.value = zeros(self.shape)
        else:
            self.value = np.array(points).reshape(*shape)


    # Some templates
    @staticmethod
    def x_cross(size, max_height=3):
        fill_diagonal(pour := zeros((size,size)), 2)
        fill_diagonal(pour2 := zeros((size,size)), 2)
        pour = pour + np.fliplr(pour2)
        pour[size//2,size//2] = 2
        return Sandpile(pour, pour.size, max_height)
        
    @staticmethod
    def t_cross(size, max_height=3):
        pour = np.zeros((size,size))
        pour[:, size//2] = pour[0, :] = 2
        return Sandpile(pour, pour.size, max_height)
        
    @staticmethod
    def Y(size, max_height=3):
        pour = x_cross(size)
        pour[size//2:,:] = 0
        pour [size//2:, size//2] = 2
        return Sandpile(pour, pour.size, max_height)
        
    @staticmethod
    def peace(size, max_height=3):
        pour = x_cross(size)
        pour[:size//2,:] = 0
        pour [:size//2, size//2] = 2
        return Sandpile(pour, pour.size, max_height)
        
            
            
    @staticmethod
    def drip(shape=(3,3), max_height=3, dtype=np.int32):
        dim = shape[0] // 2
        drop = zeros(shape, dtype=dtype)
        drop[dim,dim] = 1
        return Sandpile(drop, shape=shape, max_height=max_height, dtype=np.int32)
    
    def drip_(self):
        self.value = (self + self.drip(self.shape)).value
        return self
        
    @staticmethod
    def zeros(shape=(3,3), max_height=3, n=1, dtype=np.int32):
        if n == 1:
            return Sandpile(zeros(shape), shape, max_height)
        else:
            return [Sandpile(zeros(shape), shape, max_height) for _ in range(n)]

    def zeros_(self):
        self.value = zeros(self.shape, dtype=self.dtype)

    def clean_(self):
        self.value = zeros(self.shape)
        return self
    
    @staticmethod
    def ones(shape=(3,3), n=1, max_height=3, dtype=np.int32):
        if n == 1:
            return Sandpile(ones(shape, dtype=dtype), shape, max_height, dtype=dtype)
        else:
            return [Sandpile(ones(shape, dtype=dtype), shape) for _ in range(n)]
    
    def ones_(self):
        self.value = ones(self.shape, dtype=self.dtype)
        return self
    
    @staticmethod
    def pour(shape=(3,3), max_height=3, pours=1):
        sand = [Sandpile(randint(0, max_height, shape), shape, max_height) for _ in range(pours)]
        if len(sand) == 1:
            sand = sand[0]
        return sand
    
    def add_(self, other):
        self.value = (self + other).value
        return self
    
    def sneeze_(self):
        self.value = randint(0, self.max_height, self.shape)
        return self
    
    def watch_sand(self, drop, iterations=100, save=False, save_to=None):
        images = []
        fig = plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])
        
        for _ in range(iterations):
            im = plt.imshow(self.value, cmap=self.cmap, animated=True)
            images.append([im])
            self.add_(drop)
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        
        ani = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=500)
        if save:
            ani.save(save_to, writer=writer)
        else:
            plt.show()
    
    def fetch(self, idx):
        return self.value[idx]
    
    def show(self):
        plt.imshow(self.value, cmap=self.cmap)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        
    def __get_neighbors(self, row, col):
        # return bottom, left, top, right in form [row, column]
        dim = self.shape[0]
        neighbors = [(), ()]
        if col-1 >= 0:
            neighbors[0] += (row,)
            neighbors[1] += (col-1,)
        if row-1 >= 0:
            neighbors[0] += (row-1,)
            neighbors[1] += (col,)
        if col+1 < dim:
            neighbors[0] += (row,)
            neighbors[1] += (col+1,)
        if row+1 < dim:
            neighbors[0] += (row+1,)
            neighbors[1] += (col,)
        return neighbors
    
    def __propagate(self, res):
        height = self.max_height
        while np.any(res > height):
            for (r,c), val in np.ndenumerate(res):
                if val > height:
                    res[r,c] -= height+1
                    neighbors = self.__get_neighbors(r,c)
                    res[neighbors] += 1
        return Sandpile(res.astype(np.int32), self.shape, self.max_height)
    
    def __repr__(self):
        """Prints the """
        return repr(self.value)
    
    def __add__(self, other):
        assert isinstance(other, type(self)), \
            f"Cannot add sandpile and {type(other)}"
        res = self.value + other.value
        return self.__propagate(res)
    
    def __getitem__(self, idx):
        return self.value[idx]
    
    def __setitem__(self, idx, val):
        self.value[idx] = val
        return self
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            