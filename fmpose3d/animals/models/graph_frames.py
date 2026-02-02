import numpy as np

class Graph():
    """ The Graph to model the skeletons of human body/hand/rat/animal

    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration

        layout (string): must be one of the follow candidates
        - 'hm36_gt': Ground truth structure of Human3.6M, with 17 joints per frame
        - 'animal3d': Skeleton structure for Animal3D dataset, with 26 joints per frame
        - 'rat7m': Skeleton structure for Rat7M dataset, with 20 joints per frame

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
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop) # [17,17], adjacent=1, self=0, others=inf

        # get distance of each node to center
        self.dist_center = self.get_distance_to_center(layout)  # dist_center: distance from each node to joint 7
        self.get_adjacency(strategy)

    def get_distance_to_center(self,layout): 
        """
        :return: get the distance of each node to center
        For hm36_gt: center is joint 7
        For animal3d: center is joint 18 (neck, root joint)
        For rat7m: center is joint 4 (SpineM, root joint)
        """
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+7] = [1, 2, 3, 4, 2, 3, 4]
                dist_center[index_start+7 : index_start+11] = [0, 1, 2, 3]
                dist_center[index_start+11 : index_start+17] = [2, 3, 4, 2, 3, 4]
        elif layout == 'animal3d':
            # Animal3D: 26 joints, center is joint 18 (neck)
            # Calculate distance from each joint to neck (joint 18) along skeleton
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                # Distance from each joint to neck (18):
                # Head: 24(1)→18(0), eyes/mouth from nose, ears from eyes
                # Front legs: 18→shoulder→thigh→knee→paw
                # Hind legs: 18→tail_base→thigh→knee→paw
                # Tail: 18→tail_base→tail_mid→tail_end
                dist_center[index_start+0 : index_start+26] = [
                    2, 2, 2, 4, 4,  # 0-4: left_eye, right_eye, mouth_mid, left_front_paw, right_front_paw
                    4, 4, 1, 2, 2,  # 5-9: left_back_paw, right_back_paw, tail_base, left_front_thigh, right_front_thigh
                    2, 2, 1, 1, 3,  # 10-14: left_back_thigh, right_back_thigh, left_shoulder, right_shoulder, left_front_knee
                    3, 3, 3, 0, 3,  # 15-19: right_front_knee, left_back_knee, right_back_knee, neck(center), tail_end
                    3, 3, 3, 3, 1,  # 20-24: left_ear, right_ear, left_mouth, right_mouth, nose
                    2               # 25: tail_mid
                ]

        return dist_center

    def __str__(self):
        return self.A

    def graph_link_between_frames(self,base):
        """
        calculate graph link between frames given base nodes and seq_ind
        :param base:
        :return:
        """
        return [((front) + i*self.num_node_each, (back)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base] # Connect joints across frames.


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
            
        elif layout == 'animal3d':
            # Animal3D: 26 joints
            # Joint indices from animaldata.md:
            # 0: left_eye, 1: right_eye, 2: mouth_mid, 3: left_front_paw, 4: right_front_paw
            # 5: left_back_paw, 6: right_back_paw, 7: tail_base, 8: left_front_thigh, 9: right_front_thigh
            # 10: left_back_thigh, 11: right_back_thigh, 12: left_shoulder, 13: right_shoulder
            # 14: left_front_knee, 15: right_front_knee, 16: left_back_knee, 17: right_back_knee
            # 18: neck, 19: tail_end, 20: left_ear, 21: right_ear, 22: left_mouth, 23: right_mouth
            # 24: nose, 25: tail_mid
            # Root joint: 18 (neck - most stable central point)
            
            self.num_node_each = 26
            
            # Neighbour connections based on skeleton from animaldata.md
            neighbour_base = [
                # Head connections
                (24, 0),   # nose → left_eye
                (24, 1),   # nose → right_eye
                (1, 21),   # right_eye → right_ear
                (0, 20),   # left_eye → left_ear (corrected)
                (24, 2),   # nose → mouth_mid
                (2, 22),   # mouth_mid → left_mouth
                (2, 23),   # mouth_mid → right_mouth
                (24, 18),  # nose → neck
                
                # Upper body (neck/shoulders/front legs)
                (18, 12),  # neck → left_shoulder
                (18, 13),  # neck → right_shoulder
                (12, 8),   # left_shoulder → left_front_thigh
                (13, 9),   # right_shoulder → right_front_thigh
                (8, 14),   # left_front_thigh → left_front_knee
                (9, 15),   # right_front_thigh → right_front_knee
                (14, 3),   # left_front_knee → left_front_paw
                (15, 4),   # right_front_knee → right_front_paw
                
                # Spine and hind legs
                (18, 7),   # neck → tail_base
                (7, 10),   # tail_base → left_back_thigh
                (7, 11),   # tail_base → right_back_thigh
                (10, 16),  # left_back_thigh → left_back_knee
                (11, 17),  # right_back_thigh → right_back_knee
                (16, 5),   # left_back_knee → left_back_paw
                (17, 6),   # right_back_knee → right_back_paw
                
                # Tail
                (7, 25),   # tail_base → tail_mid
                (25, 19),  # tail_mid → tail_end
            ]
            
            # Symmetry links between left and right sides
            sym_base = [
                (0, 1),    # left_eye <-> right_eye
                (20, 21),  # left_ear <-> right_ear
                (22, 23),  # left_mouth <-> right_mouth
                (12, 13),  # left_shoulder <-> right_shoulder
                (8, 9),    # left_front_thigh <-> right_front_thigh
                (14, 15),  # left_front_knee <-> right_front_knee
                (3, 4),    # left_front_paw <-> right_front_paw
                (10, 11),  # left_back_thigh <-> right_back_thigh
                (16, 17),  # left_back_knee <-> right_back_knee
                (5, 6),    # left_back_paw <-> right_back_paw
            ]
            
            self_link, time_link = self.basic_layout(neighbour_base, sym_base)
            
            # Body parts for animal skeleton
            self.left_front = [12, 8, 14, 3]      # Left front leg
            self.right_front = [13, 9, 15, 4]     # Right front leg
            self.left_hind = [10, 16, 5]          # Left hind leg
            self.right_hind = [11, 17, 6]         # Right hind leg
            self.spine = [18, 7]                  # Neck and tail_base
            self.tail = [7, 25, 19]               # Tail
            self.head = [24, 0, 1, 2, 20, 21, 22, 23]  # Head
            self.cb = self.spine + self.tail + [12, 13]  # Center body
            
            # For compatibility with original structure
            self.la = self.left_front
            self.ra = self.right_front
            self.ll = self.left_hind
            self.rl = self.right_hind
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]
            
            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link
            
            # Center node: joint 18 (neck, root joint)
            self.center = 18
            
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
            
def get_hop_distance(num_node, edge, max_hop=1): # Build adjacency matrix; neighbors set to 1.
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]# GET [I, A]; matrix_power computes powers (0th -> identity, 1st -> A)
    arrive_mat = (np.stack(transfer_mat) > 0) # [2,17,17]
    for d in range(max_hop, -1, -1): # preserve A(i,j)=1 while A(i,i)=0; neighbors=1, diagonal=0
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
    
    # Test Animal3D skeleton
    print("\nTesting Animal3D skeleton (26 joints):")
    graph_animal = Graph('animal3d', 'spatial', 1)
    print(f"  Adjacency matrix shape: {graph_animal.A.shape}")
    print(f"  Center joint: {graph_animal.center} (neck)")
    print(f"  Number of nodes: {graph_animal.num_node}")
    print(f"  Body parts:")
    print(f"    - Left front leg: {graph_animal.left_front}")
    print(f"    - Right front leg: {graph_animal.right_front}")
    print(f"    - Left hind leg: {graph_animal.left_hind}")
    print(f"    - Right hind leg: {graph_animal.right_hind}")
    print(f"    - Head: {graph_animal.head}")
    print(f"    - Tail: {graph_animal.tail}")
    print(f"  Distance to center (joint 18): {graph_animal.dist_center}")
    
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