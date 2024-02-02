import numpy as np
import os
import pandas as pd
data_folder = "assist09"
data = np.load(os.path.join(data_folder, data_folder+'.npz'))
#['problem', 'skill_num', 'real_len', 'problem_num', 'skill', 'y']
y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']
skill_num, pro_num = data['skill_num'], data['problem_num']

df_y = pd.DataFrame(y)
df_y.to_csv('01_y.csv')

df_skill = pd.DataFrame(skill)
df_skill.to_csv('02_skill.csv')

df_problem = pd.DataFrame(problem)
df_problem.to_csv('03_problem.csv')

df_real_len = pd.DataFrame(real_len)
df_real_len.to_csv('04_real_len.csv')

# df_skill_num = pd.DataFrame(skill_num)
# df_skill_num.to_csv('05_skill_num.csv')
#
# df_pro_num = pd.DataFrame(pro_num)
# df_pro_num.to_csv('06_pro_num.csv')



# embed data, used for initialize
embed_data = np.load(os.path.join(data_folder, 'embedding_200.npz'))
_, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
df_pre_pro_embed = pd.DataFrame(pre_pro_embed)
df_pre_pro_embed.to_csv('2_0.pre_pro_embed.csv')
