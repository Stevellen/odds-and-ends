# %%
import numpy as np
from numpy import zeros, ones, product
from numpy.random import randint
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# %%
class Sandpile:
    def __init__(self, points=None, shape=(3,3), max_height=3):
        self.shape = shape
        self.max_height = max_height
        self.dtype = np.int32
        if points is None:
            self.value = np.random.randint(0,4, shape)
        else:
            self.value = np.array(points).reshape(*shape)
            
            
    @staticmethod
    def drip(shape=(3,3), dtype=np.int32):
        dim = shape[0] // 2
        drop = np.zeros(shape, dtype=dtype)
        drop[dim,dim] = 1
        return Sandpile(drop, shape)
    
    def drip_(self):
        return self.drip(self.shape)
        
    @staticmethod
    def zeros(shape=(3,3), n=1, dtype=np.int32):
        if n == 1:
            return Sandpile(zeros(shape), shape)
        else:
            return [Sandpile(zeros(shape, dtype=dtype), shape) for _ in range(n)]

    def clean_(self):
        self.value = zeros(self.shape)
        return self
    
    @staticmethod
    def ones(shape=(3,3), n=1, dtype=np.int32):
        if n == 1:
            return Sandpile(ones(shape, dtype=dtype), shape)
        else:
            return [Sandpile(ones(shape, dtype=dtype), shape) for _ in range(n)]
    
    def ones_(self):
        self.value = ones(self.shape, dtype=self.dtype)
        return self
    
    @staticmethod
    def pour(shape=(3,3), pours=1, max_height=3):
        sand = [Sandpile(randint(0,4, shape), shape, max_height) for _ in range(pours)]
        if len(sand) == 1:
            sand = sand[0]
        return sand
    
    def add(self, other, other_other=None):
        if other_other is None:
            return self + other
        return other + other_other
    
    def sneeze_(self):
        self.value = randint(0, 4, self.shape)
        return self
    
    @staticmethod
    def watch_sand(pile=None, shape=(3,3), iterations=100):
        if not isinstance(pile, Sandpile):
            pile = Sandpile.zeros(shape)
        for _ in range(iterations):
            plt.xticks([])
            plt.yticks([])
            plt.imshow(pile.value)
            pile = pile.add(pile.drip(shape))
            plt.pause(0.1)
        plt.show()
    
    def fetch(self, idx):
        return self.value[idx]
        
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
        return Sandpile(res.astype(np.int32).flatten(), self.shape)
    
    def __repr__(self):
        return repr(self.value)
    
    def __add__(self, other):
        assert isinstance(other, type(self)), \
            f"Cannot add sandpile and {type(other)}"
        res = self.value + other.value
    
    def __getitem__(self, idx):
        return self.value[idx]
    
    def __setitem__(self, idx, val):
        self.value[idx] = val
        return self
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            