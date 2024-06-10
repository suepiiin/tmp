#%%
import numpy as np
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import random
#%%
# グラフ作成

def create_custom_graph(num_nodes, num_edges_per_node):
    """指定された数のノードと各ノードが近くのノードに指定された数のエッジを持つグラフを作成します。"""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for node in G.nodes:
        # 既存のエッジを考慮してランダムにエッジを追加
        neighbors = set(G.neighbors(node))
        while len(neighbors) < num_edges_per_node:
            potential_neighbor = random.randint(0, num_nodes - 1)
            if potential_neighbor != node and potential_neighbor not in neighbors:
                G.add_edge(node, potential_neighbor)
                neighbors.add(potential_neighbor)

    # 各ノードに人口変数を割り当てる
    population_values = {node: random.randint(1000, 10000) for node in G.nodes}
    nx.set_node_attributes(G, population_values, 'population')


    return G

# グラフの作成
num_nodes = 20
num_edges_per_node = 3
G = create_custom_graph(num_nodes, num_edges_per_node)

# 結果の表示
print("Nodes:", G.nodes)
print("Edges:", G.edges)

# グラフを可視化する
pos = nx.spring_layout(G)  # ノードの位置を計算
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue')  # グラフを描画

# 各ノードのラベルの下に人口を表示
for node, (x, y) in pos.items():
    plt.text(x, y-0.1, s=str(G.nodes[node]['population']) + ' people', horizontalalignment='center')

plt.show()

#%%
# 連結な部分グラフを列挙
def generate_connected_subgraphs(G, node, visited, current_subgraph, all_subgraphs):
    visited.add(node)
    current_subgraph.add(node)
    
    # ソートしてタプルに変換し、イミュータブルな形にして保存
    subgraph_tuple = tuple(sorted(current_subgraph))
    if subgraph_tuple not in all_subgraphs:
        all_subgraphs.add(subgraph_tuple)
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                generate_connected_subgraphs(G, neighbor, visited, current_subgraph, all_subgraphs)
    
    current_subgraph.remove(node)
    visited.remove(node)

def enumerate_connected_subgraphs_recursive(G):
    all_subgraphs = set()
    for node in G.nodes:
        visited = set()
        current_subgraph = set()
        generate_connected_subgraphs(G, node, visited, current_subgraph, all_subgraphs)
    return all_subgraphs

# 使用例
connected_subgraphs = enumerate_connected_subgraphs_recursive(G)
result = [list(subgraph) for subgraph in connected_subgraphs]
print("連結な部分グラフの頂点集合:", result)


#%%
population_list = list(nx.get_node_attributes(G, 'population').values())
population_list

def create_incidence_matrix(result, num_nodes):
    matrix = np.zeros((num_nodes, len(result)), dtype=int)    
    for j, subgraph in enumerate(result):
        for node in subgraph:
            matrix[node][j] = 1
    return matrix

a = create_incidence_matrix(result, num_nodes)

#%%
# 問題のパラメータ（例として仮定したデータを使用）
m = num_nodes  # 市区町村の数
n = len(result)   # 選挙区の数
k = 3   # 選ばれる選挙区の数
a = a  # 例としての隣接行列
q = sum((a.T * population_list).T)  # 各市区町村の人口

M = max(q) + 1  # Mは十分に大きな定数

# 問題の定義
prob = pulp.LpProblem("Political_Districting", pulp.LpMinimize)

# 決定変数
x = pulp.LpVariable.dicts("x", range(n), cat='Binary')
u = pulp.LpVariable("u", lowBound=0, cat='Continuous')
l = pulp.LpVariable("l", lowBound=0, cat='Continuous')

# 目的関数
prob += u - l

# 制約条件
#for j in range(n):
#    prob += pulp.lpSum(a[i][j] * x[j] for i in range(m)) == 1
for i in range(m):
   prob += pulp.lpSum(a[i][j] * x[j] for j in range(n)) == 1

#for j in range(n):
##    prob += q[j] * x[j] <= u
#    prob += q[j] * x[j] + M * (1 - x[j]) >= l

for j in range(n):
    prob += q[j] * x[j] <= u
    prob += q[j] * x[j] + M * (1 - x[j]) >= l

prob += pulp.lpSum(x[j] for j in range(n)) == k

# 問題を解く
prob.solve()

# 結果の出力
print("Status:", pulp.LpStatus[prob.status])
print("選ばれた選挙区:")
for j in range(n):
    if pulp.value(x[j]) == 1:
        print(f"選挙区 {j} が選ばれ、人口は q[j] です")

print(f"最大人口 (u): {pulp.value(u)}")
print(f"最小人口 (l): {pulp.value(l)}")

# %%

# %%

# %%

# %%


# %%

# %%
print(a.T[54])
print(a.T[56])
print(a.T[76])
# %%

# %%
q
# %%
sum(a.T[54] * population_list)
# %%
