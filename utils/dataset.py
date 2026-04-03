import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from utils.audio_features import extract_mfcc, extract_temporal_features, extract_energy_features

# （为了节省篇幅，这里放入你原本代码中的 MFCCDataset, MFCCTemporalDataset, MFCCEnergyDataset, ThreeFeatureDataset 这四个类的完整代码，逻辑完全不用改，只需在文件开头加上上面的 import 即可）
# ... [把那四个 Dataset 类的代码粘贴到这里] ...
