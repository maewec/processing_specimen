import copy
import os
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import linregress
import itertools
from scipy import interpolate
from PIL import Image, ImageOps

from padmne.pdnforcrack.forcrack import OneCycle

plt.rcParams['figure.facecolor'] = (1,1,1,1)


# цветовые комбинации для графиков
COLORS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1),
          (1, 0, 1), (1, 1, 0), (0, 0, 0.5), (0, 0.5, 0), (0, 0.5, 0.5),
          (0, 0.5, 1), (0, 1, 0.5), (0.5, 0, 0), (0.5, 0, 0.5), (0.5, 0, 1),
          (0.5, 0.5, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 1), (0.5, 1, 0), (0.5, 1, 0.5),
          (0.5, 1, 1), (1, 0, 0.5), (1, 0.5, 0), (1, 0.5, 0.5), (1, 0.5, 1),
          (1, 1, 0.5)]

