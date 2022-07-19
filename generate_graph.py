#%%
import networkx as nx
from ortoolpy import networkx_draw #グラフの描画に使用
import matplotlib.pyplot as plt    #グラフの描画に使用
import random

import xml.etree.ElementTree as ET
# 元のXMLファイルに合ったnamespaceを設定する
ET.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")
ET.register_namespace('', "http://graphml.graphdrawing.org/xmlns")
ET.register_namespace('schemaLocation', "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")

#%%

# taskを生成
def create_task_base(agent_num):
    l_start_id = []
    l_goal_id = []
    for _ in range(agent_num):
        l_start_id.append(random.randint(0,agent_num))
        l_goal_id.append(random.randint(0,agent_num))

    return l_start_id, l_goal_id

# mapを生成
def create_map_base(node_num, edge_prob, l_start_id, vis=False):
    while True:
        g = nx.fast_gnp_random_graph(node_num, edge_prob, directed=False)
        if nx.is_connected(g):
            break
    #pos = networkx_draw(g, nx.spring_layout(g))
    pos = nx.spring_layout(g)

    cap = []
    for i in range(len((g.nodes()))):
        cap.append(1 if random.random() < cap_prob else 0)

    for start_id in l_start_id:
        cap[start_id] = 1

    l_pos_i=[]
    l_pos_j=[]
    for i in range(len(pos)):
        pos_i = int((pos[i][0] + 1.0) * 100 / 2)
        pos_j = int((pos[i][1] + 1.0) * 100 / 2)

        l_pos_i.append(pos_i)
        l_pos_j.append(pos_j)

    l_edge_start = []
    l_edge_goal = []
    for edge in list(g.edges()):
        l_edge_start.append(edge[0])
        l_edge_goal.append(edge[1])

    if vis == True:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        for dge_start, edge_goal in zip(l_edge_start, l_edge_goal):
            p1 = (pos[dge_start] + 1.0) * 100 / 2
            p2 = (pos[edge_goal] + 1.0) * 100 / 2
            ax.plot([int(p1[0]),int(p2[0])], [int(p1[1]),int(p2[1])],color="gray")

        ax.scatter(l_pos_i, l_pos_j, c=cap)
        plt.xlim(-10,110)
        plt.ylim(-10,110)
        plt.show()    

    return l_pos_i, l_pos_j, l_edge_start, l_edge_goal, cap

# %%
# xmlの整形を実行
def _pretty_print(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        _pretty_print(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + ('\t' * depth)
        else:
            parent[index - 1].tail = '\n' + ('\t' * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + ('\t' * (depth - 1))

# taskをxml形式に変換
def create_task_xml(inpath, outpath, task_data):

    l_start_id, l_goal_id = task_data
    template_path = inpath
    with open(template_path,'rt') as f:
        tree = ET.ElementTree()
        tree.parse(f)
    root = tree.getroot()
    graph = root

    for ni, (start_id, goal_id) in enumerate(zip(l_start_id, l_goal_id)):
        new_agent = ET.Element("agent")
        new_agent.set("start_id", str(start_id))
        new_agent.set("goal_id", str(goal_id))
        graph.insert(ni, new_agent)

    _pretty_print(root)
    tree = ET.ElementTree(root)
    tree.write(outpath) # 書き込み

# mapをxml形式に変換
def create_map_xml(inpath, outpath, map_data):

    l_pos_i, l_pos_j, l_cap, l_edge_start, l_edge_goal = map_data
    template_path = inpath
    with open(template_path,'rt') as f:
        tree = ET.ElementTree()
        tree.parse(f)
    root = tree.getroot()
    graph = root[2] # keyではなくgraphを選択

    for ni, (pos_i, pos_j, cap) in enumerate(zip(l_pos_i, l_pos_j, l_cap)):
        new_node = ET.Element("node")
        new_node.set("id", "n"+str(ni))

        new_data = ET.SubElement(new_node, 'data')
        new_data.set('key','key0')
        new_data.text = str(pos_i) + ", " + str(pos_j) + ", " + str(cap)
        graph.insert(ni, new_node)

    for ei, (start_node, goal_node) in enumerate(zip(l_edge_start, l_edge_goal)):
        id = 2 * ei + len(l_pos_i)
        new_edge = ET.Element("edge")
        new_edge.set("id", "e"+str(2 * ei))
        new_edge.set("source", "n"+str(start_node))
        new_edge.set("target", "n"+str(goal_node))

        new_data = ET.SubElement(new_edge, 'data')
        new_data.set('key','key1')
        new_data.text = "1"
        graph.insert(id, new_edge)

        new_edge = ET.Element("edge")
        new_edge.set("id", "e"+str(2 * ei + 1))
        new_edge.set("target", "n"+str(start_node))
        new_edge.set("source", "n"+str(goal_node))

        new_data = ET.SubElement(new_edge, 'data')
        new_data.set('key','key1')
        new_data.text = "1"
        graph.insert(id+1, new_edge)

    _pretty_print(root)
    tree = ET.ElementTree(root)
    tree.write(outpath) # 書き込み
# %%

#エージェントの数
agent_num = random.randint(2, 5)
#ノードの数
node_num = random.randint(10, 20)
#辺ができる確率
edge_prob = random.randint(10, 20)/100
# すれ違い可能なノードの発生確率
cap_prob = random.randint(10, 20)/100

# taskをランダム生成
l_start_id, l_goal_id = create_task_base(agent_num)
# mspをランダム生成
l_pos_i, l_pos_j, l_edge_start, l_edge_goal, l_cap = create_map_base(node_num, edge_prob, l_start_id, True)

# %%
# taskをxml形式に変換
task_data = [l_start_id, l_goal_id]
inpath = "./_template_task.xml"
outpath = "./out_task.xml"
create_task_xml(inpath, outpath, task_data)
# %%
# mapをxml形式に変換
map_data = [l_pos_i, l_pos_j, l_cap, l_edge_start, l_edge_goal]
inpath = "./_template_map.xml"
outpath = "./out_map.xml"
create_map_xml(inpath, outpath, map_data)


#%%

# %%
