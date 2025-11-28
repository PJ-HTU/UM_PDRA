import torch
import numpy as np
import random
import networkx as nx

###################################################################################################
# If problem_type = 'unified': trained on 33% drone_l, 33% drone_ltw, 33% drone_lo
# problem_type can be drone_l, drone_ltw, drone_lo and their any combinations, e.g., drone_ltwo
# Where drone_l is for PDRA-Basic, drone_o is for PDRA-OR, drone_tw is for PDRA-TW in this paper

def get_random_problems(batch_size, problem_size, original_node_count, link_count, problem_type):
    train_s, train_d, adj_matrices = generate_network_batch(batch_size, problem_size,
                                                            original_node_count, link_count)
    depot_xy = train_s[:, 0:1, :] 
    node_xy = train_s[:, 1:, :]
    node_demand = train_d[:, 1:, :].squeeze()

###################################################################################################
    node_serviceTime = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    node_lengthTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    node_earlyTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)
    # default velocity = 1.0

    node_lateTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    route_open = torch.zeros(size=(batch_size, 1))
    # shape: (batch, 1)

    seed_for_problem_tw = np.random.rand()
    seed_for_problem_o = np.random.rand()
    
    if ((problem_type == 'unified' and seed_for_problem_tw >= 0.5) or 'tw' in problem_type): 
        # problem_type is 'unified' or there is 'TW' in the problem_type 

        node_serviceTime = torch.rand(size=(batch_size, problem_size)) * 0.05 +0.15
        # shape: (batch, problem)
        # range: (0.15,0.2) for T=4.6 

        node_lengthTW = torch.rand(size=(batch_size, problem_size)) * 0.05 +0.15
        # shape: (batch, problem)
        # range: (0.15,0.2) for T=4.6 

        d0i = ((node_xy - depot_xy.expand(size=(batch_size,problem_size,2)))**2).sum(2).sqrt()
        # shape: (batch, problem)

        ei = torch.rand(size=(batch_size, problem_size)).mul((torch.div((4.6*torch.ones(size=(batch_size, problem_size))
                        - node_serviceTime - node_lengthTW),d0i) - 1)-1)+1
        # shape: (batch, problem)
        # default velocity = 1.0

        node_earlyTW = ei.mul(d0i)
        # shape: (batch, problem)
        # default velocity = 1.0

        node_lateTW = node_earlyTW + node_lengthTW
        # shape: (batch, problem)

    if ((problem_type == 'unified' and seed_for_problem_o >= 0.5) or 'o' in problem_type): 
        # problem_type is 'unified' or there is 'O' in the problem_type 
        route_open = torch.ones(size=(batch_size, 1))
        # shape: (batch, 1)   

    return depot_xy, node_xy, node_demand, adj_matrices, node_lateTW, route_open

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

class GridNetworkGenerator:
    def __init__(self, node_count: int, link_count: int):
        """Initialize grid network generator"""
        self.node_count = node_count  # Original node count (including depot)
        self.link_count = link_count
        self.grid_size = int(np.ceil(np.sqrt(node_count - 1)))  # Grid size excluding depot

    def generate_initial_grid(self) -> tuple:
        """Generate initial grid nodes (depot is ID=0, other nodes start from 1, no depot connections added)"""
        nodes = {}
        edges = []
        offset_range = 1 / (2 * (self.grid_size - 1))
        
        # Generate random depot node (ID=0)
        depot_x = random.uniform(0, 1)
        depot_y = random.uniform(0, 1)
        nodes[0] = (depot_x, depot_y)
        
        # Generate other original nodes (ID starts from 1)
        node_id = 1
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if node_id < self.node_count:
                    x = j / (self.grid_size - 1)
                    y = i / (self.grid_size - 1)
                    
                    x += random.uniform(-offset_range, offset_range)
                    y += random.uniform(-offset_range, offset_range)
                    
                    x = max(0, min(1, x))
                    y = max(0, min(1, y))
                    
                    nodes[node_id] = (x, y)
                    node_id += 1
        
        # Generate grid edges (excluding depot connections, only connections between original nodes)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current_id = i * self.grid_size + j + 1
                if current_id >= self.node_count:
                    break
                if j < self.grid_size - 1:
                    right_id = i * self.grid_size + (j + 1) + 1
                    if right_id < self.node_count:
                        edges.append((current_id, right_id))
                if i < self.grid_size - 1:
                    down_id = (i + 1) * self.grid_size + j + 1
                    if down_id < self.node_count:
                        edges.append((current_id, down_id))
        
        return nodes, edges

    def prune_edges(self, edges: list, nodes: int) -> list:
        """Prune edges"""
        
        # Build minimum spanning tree
        G = nx.Graph(edges)
        if not nx.is_connected(G):
            raise ValueError("Graph is not connected, cannot prune edges")
        
        # Calculate actual node count (original node count excluding depot)
        actual_node_count = nodes - 1  # depot is ID=0, other original nodes count is nodes-1
        min_edges = actual_node_count - 1  # Minimum edge count for connected graph
        
        # Intelligently select starting node (prioritize nodes in middle region)
        # Assuming node IDs are 0 to nodes-1, select nodes in middle region
        center_range = max(1, int(nodes * 0.25))  # Define middle region range
        center_nodes = list(range(center_range, nodes - center_range))
        
        if center_nodes:
            # Prioritize selecting nodes from middle region
            start_node = random.choice(center_nodes)
        else:
            # If middle region is empty, fall back to fully random selection
            start_node = random.choice(list(G.nodes()))
        
        # BFS to build spanning tree
        visited = {start_node}
        tree_edges = []
        queue = [start_node]
        
        while queue and len(tree_edges) < min_edges:
            node = queue.pop(0)
            neighbors = list(G.neighbors(node))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    tree_edges.append((node, neighbor))
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(tree_edges) == min_edges:
                        break
        
        # Add additional edges
        need_add = self.link_count - min_edges
        
        all_edges = set(edges)
        tree_set = set([(u, v) for u, v in tree_edges]) | set([(v, u) for u, v in tree_edges])
        non_tree_edges = [e for e in all_edges if e not in tree_set]
        
        return tree_edges + random.sample(non_tree_edges, need_add)
        
    def split_edges(self, nodes: dict, edges: list) -> tuple:
        all_nodes = nodes.copy()
        new_edges = []
        next_id = max(nodes.keys()) + 1
        
        for a, b in edges:
            xa, ya = nodes[a]
            xb, yb = nodes[b]
            xc, yc = (xa+xb)/2, (ya+yb)/2
            
            # Calculate perpendicular offset
            length = np.sqrt((xb-xa)**2 + (yb-ya)**2)
            if length > 0:
                perp_x = -(yb-ya)/length
                perp_y = (xb-xa)/length
                dev = random.uniform(-length*0.3, length*0.3)
                xc += perp_x * dev
                yc += perp_y * dev
            
            # Boundary reflection handling
            xc_reflected, yc_reflected = xc, yc
            
            # X direction reflection
            if xc < 0:
                xc_reflected = -xc  # Reflect about x=0
            elif xc > 1:
                xc_reflected = 2 - xc  # Reflect about x=1
            
            # Y direction reflection
            if yc < 0:
                yc_reflected = -yc  # Reflect about y=0
            elif yc > 1:
                yc_reflected = 2 - yc  # Reflect about y=1
            
            # Ensure reflected point is within boundary (handle multiple reflections)
            xc = max(0, min(1, xc_reflected))
            yc = max(0, min(1, yc_reflected))
            
            all_nodes[next_id] = (xc, yc)
            new_edges.append((a, next_id))
            new_edges.append((next_id, b))
            next_id += 1
        
        return all_nodes, new_edges

    def create_adjacency_matrix(self, nodes: dict, edges: list, original_ids: list) -> np.ndarray:
        """Generate adjacency matrix (fully connected between original nodes, depot connects to all original nodes)"""
        n = len(nodes)
        adj = np.zeros((n, n), dtype=int)
        node2idx = {node_id: i for i, node_id in enumerate(sorted(nodes.keys()))}
        
        # Process all edges (including connections between original nodes)
        for u, v in edges:
            i, j = node2idx[u], node2idx[v]
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Set full connectivity between original nodes (complete graph)
        for idx in original_ids:
            for idy in original_ids:
                if idx != idy:  # Avoid self-loops
                    adj[idx, idy] = 1
                    adj[idy, idx] = 1
        
        return adj

    def generate_demands(self, original_ids: list, new_ids: list) -> dict:
        """Generate demands (depot demand is 0)"""
        demands = {oid: 0 for oid in original_ids}
        demands.update({nid: random.uniform(1, 10)/10 for nid in new_ids})
        return demands

    def generate_single_network(self) -> tuple:
        """Generate single network"""
        original_nodes, initial_edges = self.generate_initial_grid()
        pruned_edges = self.prune_edges(initial_edges, self.node_count)
        all_nodes, final_edges = self.split_edges(original_nodes, pruned_edges)
        
        original_ids = list(original_nodes.keys())
        new_ids = set(all_nodes.keys()) - set(original_ids)
        adj_matrix = self.create_adjacency_matrix(all_nodes, final_edges, original_ids)
        demands = self.generate_demands(original_ids, new_ids)
        
        sorted_ids = sorted(all_nodes.keys())
        coords = np.array([all_nodes[oid] for oid in sorted_ids], dtype=np.float32)
        node_demands = np.array([demands[oid] for oid in sorted_ids], dtype=np.float32).reshape(-1, 1)
        
        return coords, node_demands, adj_matrix

def generate_network_batch(batch_size, problem_size, original_node_count, link_count):
    """Generate batch of networks"""   
    
    s = torch.zeros(batch_size, problem_size+1, 2, dtype=torch.float32)
    d = torch.zeros(batch_size, problem_size+1, 1, dtype=torch.float32)
    adj_matrices = torch.zeros(batch_size, problem_size+1, problem_size+1, dtype=torch.float32)
    
    for i in range(0, batch_size):
        generator = GridNetworkGenerator(original_node_count, link_count)
        coords, demands, adj = generator.generate_single_network()
        s[i] = torch.from_numpy(coords)
        d[i] = torch.from_numpy(demands)
        adj_matrices[i] = torch.from_numpy(adj)
    
    return s, d, adj_matrices