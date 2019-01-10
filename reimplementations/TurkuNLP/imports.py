from livelossplot import PlotLosses
from sklearn.model_selection import KFold
import random
import statistics
import json
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import time
import copy
import sys
from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import gensim
import gensim.models.keyedvectors as word2vec
from torch.autograd import Variable
from torch import FloatTensor
from torch.nn import Parameter, init
from constants import *

from xml.dom import minidom
import re
import networkx as nx
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
import numpy as np

import warnings
# "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings('ignore')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
