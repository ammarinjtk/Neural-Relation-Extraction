import gensim
import json
import csv
from scipy.stats import spearmanr
import math

import re
from tqdm import tqdm_notebook, tqdm

import collections
import os
import numpy as np

from dahuffman import HuffmanCodec

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import time

import glob
import pickle

import networkx as nx

torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
