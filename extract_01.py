import os
import sys
import numpy as np
from scipy import sparse
import time


data_folder = 'assist09'
#pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'pro_skill_sparse.npz'))

#lzj using numpy open file to understand the data struct
data_lzj=np.load(os.path.join(data_folder, 'pro_skill_sparse.npz'))

data01=data_lzj["row"]#19203
data02=data_lzj['shape']#array([15911,   123], dtype=int64)
data03=data_lzj["col"]#19203
data04=data_lzj["data"]#(19203,)
data05=data_lzj["format"]






