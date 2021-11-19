import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_vals = [1,2,3,4,5]
y_vals = [1,3,2,4,5]


def animate(x,y):

    plt.plot(x, y)