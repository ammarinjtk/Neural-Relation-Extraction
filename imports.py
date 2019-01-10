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
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.nn import Parameter, init
from torch import FloatTensor
from torch.autograd import Variable

import gensim.models.keyedvectors as word2vec
import gensim

from collections import OrderedDict

from tqdm import tqdm, tqdm_notebook
import sys
import copy
import time

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

import json

import statistics
import random
from sklearn.model_selection import KFold
from livelossplot import PlotLosses

# torch.cuda.manual_seed_all(SEED)
# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)

from allennlp.modules.elmo import Elmo, batch_to_ids
