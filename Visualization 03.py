import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('02_skill_small.csv')

# 创建一个空的有向图
G = nx.DiGraph()

# 遍历数据，添加节点和边
for index, row in data.iterrows():
    for i in range(200):
        class_name = row[i]
        if class_name not in G:
            G.add_node(class_name)
        G.add_edge(class_name, row[199])

# 绘制知识图谱
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
plt.show()
