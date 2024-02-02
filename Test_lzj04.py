import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from scipy import sparse
data_folder = 'assist09'
skill_skill_coo = sparse.load_npz(os.path.join(data_folder, 'pro_pro_sparse.npz'))





import numpy as np
file_path='./assist09/skill_skill_sparse.npz'
poem=np.load(file_path,allow_pickle=True)


f=open(os.path.join(data_folder, 'skill_id_dict.txt'), 'r')
skill_id_dict = eval(f.read())


embed=np.load(os.path.join(data_folder, 'embedding_200.npz'))
embed_pro,embed_skill,embed_pfinal=embed['pro_repre'], embed['skill_repre'], embed['pro_final_repre']



data_folder = "assist09"
data = np.load(os.path.join(data_folder, data_folder+'.npz'))
y, skill, problem, real_len,problem_number_lzj,skill_number_lzj = data['y'], data['skill'], data['problem'], data['real_len'],data['problem_num'],data['skill_num']
#['problem', 'skill_num', 'real_len', 'problem_num', 'skill', 'y']