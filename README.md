# 【Paddle-PGL】图神经网络学习笔记

- [【Paddle-PGL】图神经网络学习笔记](#-paddle-pgl----------)
  * [基于图的学习模型简介](#----------)
    + [什么是Graph embedding?](#---graph-embedding-)
    + [什么方法可以实现Graph embedding？](#--------graph-embedding-)
    + [什么场景需要用到Graph embedding？](#--------graph-embedding-)
    + [消息传递机制](#------)
    + [图卷积神经网络GCN](#-------gcn)
    + [图注意力网络GAT](#------gat)
    + [图神经网络模型进阶](#---------)
  * [图神经网络在药物相互作用预测问题上的文献理解](#----------------------)
  * [Citation](#citation)


* 课程链接：https://aistudio.baidu.com/aistudio/course/introduce/1956
* GPL github链接：https://github.com/PaddlePaddle/PGL

## 基于图的学习模型简介
图（graph）在这里定义为图论中表示物件与物件之间的关系的数学对象。在实际应用中常被用于表示复杂的关系网，如社交网络，网页链接以及我所在的研究领域"生物网络"。学习图神经网络的初衷就是为了了解如何通过深度学习的方法更好的挖掘生物学网络中的信息。当我们将物件及物件间的关联构建成网络（图），可以通过如图嵌入、基于图的机器学习等方法整合节点、边以及其拓扑结构上的关联，从而对所需要研究的问题进行更好的分析。在图上可以实现的分析包括研节点分类、链路预测、聚类和可视化等。在我所研究的系统生物学，基于图的学习方法已经被广泛应用在特征基因识别、药物响应预测等问题上。

### 什么是Graph embedding?

由于真实的图（或网络）通常是高维复杂的，因此研究者开发了图嵌入（graph embedding）方法对高维的图进行降维。简单来说，可以基于所研究的对象或问题构建出D维空间中的图，再将图中的节点映射到（嵌入到）d维向量空间中，这里的d维向量空间远小于图所在的D维空间，且在图中相邻的节点在向量空间中保持彼此靠近[1]。

### 什么方法可以实现Graph embedding？
* 基于因子分解的方法
<br>局部线性嵌入（Locally Linear Embedding ，LLE）
<br>拉普拉斯特征映射（Laplacian Eigenmaps）

* 基于随机游走的方法
<br>DeepWalk
<br>node2vec
<br>methpath2vec

* 基于深度学习的方法
<br>GCN
<br>GAT
<br>Graphsage
<br>ERNIESage

除了上述列出的经典算法外，还有许多能够进行graph embedding。

### 什么场景需要用到Graph embedding？
![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/application.png)

###图神经网络


![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/GNN.png)

### 消息传递机制
消息传递包括两部分：
<br>**Send**: 源节点发送消息，即边上的源节点，往目标节点发送特征
<br>**Recv**: 目标节点接受消息，即目标节点对收到的特征进行聚合
<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/message_passing.png)
### 图卷积神经网络GCN
<br>图卷积神经网络（Graph Convolutional Network, GCN）
<br>**图像卷积**:将一个像素点周围的像素按照不同的权重叠加起来。
<br>**图结构卷积**:将一个节点周围的邻居按照不同的权重叠加起来。
<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/gcn_formula.JPG)
<br>附上一个学习时手写的结构，包括了前边描述的消息传递过程，合GCN中关键公式的一步步解析：
<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/gcn_structure_handwriting.jpg)

```
import paddle.fluid.layers as L

def gcn_layer(gw, feature, hidden_size, activation, name, norm=None):
    """
    描述：通过GCN层计算新的节点表示
    输入：gw - GraphWrapper对象
         feature - 节点表示 (num_nodes, feature_size)
         hidden_size - GCN层的隐藏层维度 int
         activation - 激活函数 str
         name - GCN层名称 str
         norm - 标准化tensor float32 (num_nodes,)，None表示不标准化
    输出：新的节点表示 (num_nodes, hidden_size)
    """
        # send函数
    def send_func(src_feat, dst_feat, edge_feat):
        """
        描述：用于send节点信息。函数名可自定义，参数列表固定
        输入：src_feat - 源节点的表示字典 {name:(num_edges, feature_size)}
             dst_feat - 目标节点表示字典 {name:(num_edges, feature_size)}
             edge_feat - 与边(src, dst)相关的特征字典 {name:(num_edges, feature_size)}
        输出：存储发送信息的张量或字典 (num_edges, feature_size) or {name:(num_edges, feature_size)}
        """
        return src_feat["h"] # 直接返回源节点表示作为信息
    # send和recv函数是搭配实现的，send的输出就是recv函数的输入
    # recv函数
    def recv_func(msg):
        """
        描述：对接收到的msg进行聚合。函数名可自定义，参数列表固定
        输出：新的节点表示张量 (num_nodes, feature_size)
        """
        return L.sequence_pool(msg, pool_type='sum') # 对接收到的消息求和
    ### 消息传递机制执行过程
    # gw.send函数
    msg = gw.send(send_func, nfeat_list=[("h", feature)]) 
    """ 
    描述：触发message函数，发送消息并将消息返回
    输入：message_func - 自定义的消息函数
         nfeat_list - list [name] or tuple (name, tensor)
         efeat_list - list [name] or tuple (name, tensor)
    输出：消息字典 {name:(num_edges, feature_size)}
    """
    # gw.recv函数
    output = gw.recv(msg, recv_func)
    """ 
    描述：触发reduce函数，接收并处理消息
    输入：msg - gw.send输出的消息字典
         reduce_function - "sum"或自定义的reduce函数
    输出：新的节点特征 (num_nodes, feature_size)
    如果reduce函数是对消息求和，可以直接用"sum"作为参数，使用内置函数加速训练，上述语句等价于 \
    output = gw.recv(msg, "sum")
    """
    # 通过以activation为激活函数的全连接输出层
    output = L.fc(output, size=hidden_size, bias_attr=False, act=activation, name=name)
    return output
```



### 图注意力网络GAT
<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/gat_paddle.JPG)
 <br>GAT(Graph attention network)优势相比于GCN(等权重对邻居信息进行卷积操作)在于通过 Attention 机制，为不同节点分配不同权重。GAT可以采用单头或者多头的注意力机制[2]：
<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/gat_frompaper.JPG)
<br>具体计算时，单头的GAT[3]:
<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/gat_单头.JPG)
<br>多头GAT[3]:
<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/gat_多头.JPG)
<br>课堂示例中通过paddle-pgl实现的GAT函数：

```
from pgl.utils import paddle_helper
import paddle.fluid as fluid
import numpy 

def single_head_gat(graph_wrapper, node_feature, hidden_size, name):
    # 实现单头GAT

    def send_func(src_feat, dst_feat, edge_feat):
        ##################################
        # 按照提示一步步理解代码吧，你只需要填###的地方

        # 1. 将源节点特征与目标节点特征concat 起来，对应公式当中的 concat 符号，可能用到的 API: fluid.layers.concat
        #with fluid.dygraph.guard():
            #src_feat = fluid.dygraph.to_variable(src_feat)
            #dst_feat = fluid.dygraph.to_variable(dst_feat)
        #Wh = fluid.layers.concat([src_feat["Wh"], dst_feat["Wh"]])
        Wh = src_feat["Wh"] + dst_feat["Wh"]
       # 2. 将上述 Wh 结果通过全连接层，也就对应公式中的a^T

        alpha = fluid.layers.fc(Wh, 
                            size=1, 
                            name=name + "_alpha", 
                            bias_attr=False)

        # 3. 将计算好的 alpha 利用 LeakyReLU 函数激活，可能用到的 API: fluid.layers.leaky_relu
        alpha = fluid.layers.leaky_relu(alpha, 0.2)
        
        ##################################
        return {"alpha": alpha, "Wh": src_feat["Wh"]}
    
    def recv_func(msg):
        ##################################
        # 按照提示一步步理解代码吧，你只需要填###的地方

        # 1. 对接收到的 logits 信息进行 softmax 操作，形成归一化分数，可能用到的 API: paddle_helper.sequence_softmax
        alpha = msg["alpha"]
        norm_alpha = paddle_helper.sequence_softmax(alpha)###

        # 2. 对 msg["Wh"]，也就是节点特征，用上述结果进行加权
        output = norm_alpha * msg["Wh"]

        # 3. 对加权后的结果进行相加的邻居聚合，可能用到的API: fluid.layers.sequence_pool
        output = fluid.layers.sequence_pool(output, pool_type="sum")
        ##################################
        return output
    
    # 这一步，其实对应了求解公式当中的Whi, Whj，相当于对node feature加了一个全连接层

    Wh = fluid.layers.fc(node_feature, hidden_size, bias_attr=False, name=name + "_hidden")
    # 消息传递机制执行过程
    message = graph_wrapper.send(send_func, nfeat_list=[("Wh", Wh)])
    output = graph_wrapper.recv(message, recv_func)
    output = fluid.layers.elu(output)
    return output

def gat(graph_wrapper, node_feature, hidden_size):
    # 完整多头GAT

    # 这里配置多个头，每个头的输出concat在一起，构成多头GAT
    heads_output = []
    # 可以调整头数 (8 head x 8 hidden_size)的效果较好 
    n_heads = 8
    for head_no in range(n_heads):
        # 请完成单头的GAT的代码
        single_output = single_head_gat(graph_wrapper, 
                            node_feature, 
                            hidden_size, 
                            name="head_%s" % (head_no) )
        heads_output.append(single_output)
    
    output = fluid.layers.concat(heads_output, -1)
    return output
```


### 图神经网络模型进阶


## 图神经网络在药物相互作用预测问题上的文献理解
<br>**文章题目** Deep graph embedding for prioritizing synergistic anticancer drug combinations [4]
<br>**文章大纲**
<br>文章通过整合药物-药物协同关联网络（Drug-Drug Synergy association network, DDS)，药物-靶蛋白关联网络（Drug-Target Interaction Network，DTI）和蛋白相互作用网络（protein-protein interaction network， PPI)三个维度信息，构建了药物-蛋白质异质网络，基于改造版decagon算法对该异质网络进行学习，从而预测药物协同分数，并对网络嵌入得到的低维表示进行分析。
<br>**原文中的模型框架**
<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/example_workflow.JPG)
<br>**文章理解**

**1. 网络是如何构建的？**
<br>**DDS**: O'Neil数据集提供的38个药物在39个细胞系上的Loewe synergy分数，该数据集上协同分数的分布基本在[-60,60]之间。该作者将分值大于30的认为是协同，设为阳性samples；而剩下的设为阴性samples。
<br>**DTI**：STITCH数据库中提取的药物-靶蛋白关联。
<br>**PPI**： 整合 STRING(V11.0) 数据库和BioGRID(V3.5.174)数据库而成。
以上三个网络整合成一个二元异质网络，将药物组合协同预测转化为link prediction问题。

**2. 文章用的方法？**
<br>该文章基于decogan方法[5]，因此首先理解了一下decagon这个方法。
<br>decogan方法是18年发表在bioinformatics的文章，该方法初衷是设计用于预测药物组合处方时可能共出现的副作用，作者将药物与药物之间通过副作用连接起来$(v_{i},r,v_{j})$，加上药物-靶基因关联合蛋白相互作用关联共同构成一个异质网络，其中$(v_{i},r,v_{j})$有964种。对于网络中的药物节点，以只在单药使用时出现的副作用$x_{i}$作为节点特征。该异质网络中的每个节点$v_{i}$以$x_{i}$为特征通过一个encoder表示为d维的$z_{i}$，再通过一个encoder计算出r。这里的encoder是一个“transformation-aggregration”结构；decoder是采用了rank-d DEDICOM张量分解的方法直接输出目标值。这里的loss用的cross entropy。
<br>那这个方法怎么用在协同分值的预测问题上呢？其实作者基本就是采用了decagon的思路，采用了“encoder-decoder”结构，对所构建的异质网络中的节点（v_i，特征为x_i）通过4层GCN（每层间加上一个ReLu激活函数，进入decoder前再通过一个sigmoid函数）表示为z_i，再通过decoder形成输出（$s(v_{i}, v_{j})$)。
<br>比较有趣的是作者进行了cell line specific的预测。我的理解是他们将不同细胞系的协同情况定义了不同的(v_i,r,v_j)，类似于decagon中对不同的side effect进行预测，将r定义为不同细胞系中的关联关系R集合（不确定自己理解是否正确）。

**3. 该算法的performance？**

<br>![image](https://github.com/VeronicaFung/paddle_pgl_note/blob/main/dpl/performance.JPG)

**4. 其他分析？**
<br>分析了embedding是否能保留药物之间interdependent的关联。

## Citation
<br>[1] 图神经网络之图卷积网络——GCN (https://blog.csdn.net/zbp_12138/article/details/110246797)
<br>[2] Veličković P, Cucurull G, Casanova A, et al. Graph attention networks[J]. arXiv preprint arXiv:1710.10903, 2017.
<br>[3] GAT学习笔记 (https://ai.baidu.com/forum/topic/show/972764)
<br>[4] Jiang P, Huang S, Fu Z, et al. Deep graph embedding for prioritizing synergistic anticancer drug combinations[J]. Computational and Structural Biotechnology Journal, 2020, 18: 427-438.
<br>[5] Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy side effects with graph convolutional networks[J]. Bioinformatics, 2018, 34(13): i457-i466.
