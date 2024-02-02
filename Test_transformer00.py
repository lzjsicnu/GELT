import numpy as np
file_path='./assist09/skill_skill_sparse.npz'
poem=np.load(file_path,allow_pickle=True)

poem.files#['row', 'shape', 'col', 'format', 'data']
p_row,p_shape,p_col,p_data,p_fomat=poem['row'],poem['shape'],poem['col'],poem['data'],poem['format']

#是skill之间的关联！
for i in range(p_row.shape[0]):
    print("i=",i,[p_row[i]],end="\n")

    
