import numpy as np
file_path='./assist09/pro_feat.npz'
poem=np.load(file_path,allow_pickle=True)
poem.files
#只有['pro_feat']
p_feat=poem['pro_feat']
p_feat.shape
#(15911, 7)
