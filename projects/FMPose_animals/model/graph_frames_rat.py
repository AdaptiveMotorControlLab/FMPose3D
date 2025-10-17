import numpy as np

class Graph():
    """ The Graph to model the skeletons of human body/hand/rat

    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration

        layout (string): must be one of the follow candidates
        - 'hm36_gt' same with ground truth structure of human 3.6, with 17 joints per frame
        - 'rat7m' skeleton structure for Rat7M dataset, with 20 joints per frame

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout, 
                 strategy,
                 pad=0,
                 max_hop=1,
                 dilation=1):

        self.max_hop = max_hop # 1
        self.dilation = dilation # 1
        self.seqlen = pad  # 1
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop) # [17,17], 相邻位1，本身为0，其他为inf

        # get distance of each node to center
        self.dist_center = self.get_distance_to_center(layout)  # dist_center 各个节点到joint7的距离s
        self.get_adjacency(strategy)

    def get_distance_to_center(self,layout): 
        """
        :return: get the distance of each node to center
        For hm36_gt: center is joint 7
        For rat7m: center is joint 4 (SpineM, root joint)
        """
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+7] = [1, 2, 3, 4, 2, 3, 4]
                dist_center[index_start+7 : index_start+11] = [0, 1, 2, 3]
                dist_center[index_start+11 : index_start+17] = [2, 3, 4, 2, 3, 4]
        elif layout == 'rat7m':
            # Rat7M: 20 joints, center is joint 4 (SpineM)
            # Distance from each joint to joint 4 along the skeleton chain
            # parents=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                # [Joint0-3: distance going up from 4, Joint4: center=0, Joint5-19: distance going down from 4]
                dist_center[index_start+0 : index_start+20] = [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        return dist_center

    def __str__(self):
        return self.A

    def graph_link_between_frames(self,base):
        """
        calculate graph link between frames given base nodes and seq_ind
        :param base:
        :return:
        """
        return [((front) + i*self.num_node_each, (back)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base] # 把每一帧的关节点都连接起来


    def basic_layout(self,neighbour_base, sym_base):
        """
        for generating basic layout time link selflink etc.
        neighbour_base: neighbour link per frame
        sym_base: symmetrical link(for body) or cross-link(for hand) per frame

        :return: link each node with itself
        """
        self.num_node = self.num_node_each * self.seqlen
        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1) # for single frame, this is null
                     for j in range(self.num_node_each)]
        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in
                                  range(self.seqlen - 1) 
                                  for j in range(self.num_node_each)]
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in
                               range(self.seqlen - 1)
                               for j in range(self.num_node_each)]

        self_link = [(i, i) for i in range(self.num_node)]

        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)

        self.sym_link_all = self.graph_link_between_frames(sym_base)

        return self_link, time_link

    def get_edge(self, layout):
        """
        get edge link of the graph
        la,ra: left/right arm (for rat: left/right front leg)
        ll/rl: left/right leg (for rat: left/right hind leg)
        cb: center bone (spine)
        """
        if layout == 'hm36_gt':
            self.num_node_each = 17

            neighbour_base = [(0, 1), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), 
                              (7, 0), (8, 7), (9, 8), (10, 9), (11, 8),
                              (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)
                              ]
                        
            sym_base = [(6, 3), (5, 2), (4, 1), (11, 14), (12, 15), (13, 16)]  

            self_link, time_link = self.basic_layout(neighbour_base, sym_base) # self_link: node itself; time_link: 

            self.la, self.ra =[11, 12, 13], [14, 15, 16] # left and right arm
            self.ll, self.rl = [4, 5, 6], [1, 2, 3] # left and right leg
            self.cb = [0, 7, 8, 9, 10] # center bone
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link # len=39

            # center node of body/hand
            self.center = 8 - 1
            
        elif layout == 'rat7m':
            # Rat7M: 20 joints
            # Joint names and structure based on parents array
            # parents=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            # Root joint: 4 (SpineM)
            # Left: [8, 10, 11, 17, 18] (HipL, ElbowL, ArmL, KneeL, ShinL)
            # Right: [9, 14, 15, 16, 19] (HipR, ElbowR, ArmR, KneeR, ShinR)
            
            self.num_node_each = 20

            # Neighbour connections based on parent-child relationships
            # Edge (child, parent) for all joints except root (parent=-1)
            neighbour_base = [
                (1, 0), (2, 1), (3, 2), (4, 3),     # Spine chain: 0->1->2->3->4
                (5, 4), (6, 5), (7, 6),             # Continue from SpineM(4): 4->5->6->7
                (8, 7), (9, 8), (10, 9), (11, 10),  # Chain: 7->8->9->10->11
                (12, 11), (13, 12), (14, 13),       # Chain: 11->12->13->14
                (15, 14), (16, 15), (17, 16),       # Chain: 14->15->16->17
                (18, 17), (19, 18)                  # Chain: 17->18->19
            ]
            
            # Symmetry links between left and right sides
            # Left: [8, 10, 11, 17, 18], Right: [9, 14, 15, 16, 19]
            sym_base = [
                (8, 9),    # HipL <-> HipR
                (10, 14),  # ElbowL <-> ElbowR
                (11, 15),  # ArmL <-> ArmR
                (17, 16),  # KneeL <-> KneeR
                (18, 19)   # ShinL <-> ShinR
            ]

            self_link, time_link = self.basic_layout(neighbour_base, sym_base)

            # Body parts for rat skeleton
            self.left_front = [10, 11]        # Left front leg (ElbowL, ArmL)
            self.right_front = [14, 15]       # Right front leg (ElbowR, ArmR)
            self.left_hind = [17, 18]         # Left hind leg (KneeL, ShinL)
            self.right_hind = [16, 19]        # Right hind leg (KneeR, ShinR)
            self.spine = [0, 1, 2, 3, 4, 5, 6, 7]  # Spine chain
            self.hips = [8, 9]                # Hip joints
            self.cb = self.spine + self.hips + [12, 13]  # Center body
            
            # For compatibility with original structure
            self.la = self.left_front
            self.ra = self.right_front
            self.ll = self.left_hind
            self.rl = self.right_hind
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link

            # Center node: joint 4 (SpineM, root joint)
            self.center = 4
            
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation) # [0, 1]
        adjacency = np.zeros((self.num_node, self.num_node)) 
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency) 

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                a_sym = np.zeros((self.num_node, self.num_node))
                a_forward = np.zeros((self.num_node, self.num_node))
                a_back = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop: # 0:diagonal; 1:adjacent point
                            if (j,i) in self.sym_link_all or (i,j) in self.sym_link_all: # symmetrical node
                                a_sym[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.time_link_forward:
                                a_forward[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.time_link_back:
                                a_back[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] == self.dist_center[i]: 
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] > self.dist_center[i]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i] 

                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_close)
                    A.append(a_further)
                    A.append(a_sym)
                    if self.seqlen > 1: 
                        A.append(a_forward)
                        A.append(a_back)

            A = np.stack(A)
            self.A = A

        else:
            raise ValueError("Do Not Exist This Strategy")
            
def get_hop_distance(num_node, edge, max_hop=1): # 建立邻接矩阵,相邻则置0   
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]# GET [I,A]; matrix_power计算矩阵次方  0次方对角线全1，1次方不动
    arrive_mat = (np.stack(transfer_mat) > 0) # [2,17,17]
    for d in range(max_hop, -1, -1): # preserve A(i,j) = 1 while A(i,i) = 0  相邻为1 对角为0
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0) 
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node)) # 17,17 
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

if __name__=="__main__":
    # Test Human3.6M skeleton
    print("Testing Human3.6M skeleton (17 joints):")
    graph_h36m = Graph('hm36_gt', 'spatial', 1)
    print(f"  Adjacency matrix shape: {graph_h36m.A.shape}")
    print(f"  Center joint: {graph_h36m.center}")
    print(f"  Number of nodes: {graph_h36m.num_node}")
    
    # Test Rat7M skeleton
    print("\nTesting Rat7M skeleton (20 joints):")
    graph_rat = Graph('rat7m', 'spatial', 1)
    print(f"  Adjacency matrix shape: {graph_rat.A.shape}")
    print(f"  Center joint: {graph_rat.center}")
    print(f"  Number of nodes: {graph_rat.num_node}")
    print(f"  Body parts:")
    print(f"    - Left front leg: {graph_rat.left_front}")
    print(f"    - Right front leg: {graph_rat.right_front}")
    print(f"    - Left hind leg: {graph_rat.left_hind}")
    print(f"    - Right hind leg: {graph_rat.right_hind}")
    print(f"    - Spine: {graph_rat.spine}")
    print(f"  Distance to center (joint 4): {graph_rat.dist_center}")