from dataclasses import dataclass
import torch

from PDRAProblemDef import get_random_problems, augment_xy_data_by_8_fold

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    
    node_lateTW: torch.Tensor = None
    # shape: (batch, problem)
    route_open: torch.Tensor = None
    # shape: (batch, 1)

    # Used for attribute normalization
    attribute_tw: bool = False
    attribute_o: bool = False
    

# Reset_State variables to be filled for encoder use

@dataclass
class Step_State:
    
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)

    # Used for decoder query
    current_vehicle: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    current_depot_xy: torch.Tensor = None
    # shape: (batch, pomo, 2)

    collected_demand: torch.Tensor = None
    # shape: (batch, pomo)

# Step_State variables to be filled for decoder use

class PDRAEnv:
    def __init__(self, **env_params):
    
    # PDRAEnv __init__ variables to be filled
        # Const @INIT
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.problem_type = env_params['problem_type']
       
        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_adj_matrices = None
        self.saved_node_lateTW = None
        self.saved_route_open = None
        self.saved_index = None

        # Const @Load_Problem
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        
        self.node_lateTW = None
        # shape: (batch, problem, 1)

        self.attribute_tw = False
        self.attribute_o = False

        # Dynamic-1
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # Used for encoder's initial_embedding
        self.num_vehicles = None 
        self.vehicle_capacity = None
        self.current_vehicle_config = None
        self.route_open = None

        # Used for decoder's query vector
        self.current_vehicle = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.current_depot_xy = None
        self.collected_demand = None

        # Used for masking
        self.adj_matrices = None
        # shape: (batch, problem+1, problem+1)
        self.original_node_count = env_params['original_node_count']
        self.link_count = env_params['link_count']

        # states to return
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_adj_matrices = loaded_dict['adj_matrices']
        self.saved_node_lateTW = loaded_dict['node_lateTW']
        self.saved_route_open = loaded_dict['route_open']
        self.saved_index = 0

    def load_problems(self, batch_size, vehicle_config=None, aug_factor=1):
        
        # Const @Load_Problem and used for encoder's initial_embedding
        self.batch_size = batch_size
        self.num_vehicles = vehicle_config['num_vehicles']
        self.vehicle_capacity = vehicle_config['vehicle_capacity']
        self.current_vehicle_config = vehicle_config
    
        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand, adj_matrices, node_lateTW, route_open = get_random_problems(
                                                            batch_size, self.problem_size, 
                                                            original_node_count=self.original_node_count,
                                                            link_count=self.link_count,
                                                            problem_type=self.problem_type)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            adj_matrices = self.saved_adj_matrices[self.saved_index:self.saved_index+batch_size]
            node_lateTW = self.saved_node_lateTW[self.saved_index:self.saved_index+batch_size]
            route_open = self.saved_route_open[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                adj_matrices = adj_matrices.repeat(8, 1, 1)
                node_lateTW = node_lateTW.repeat(8, 1)
                route_open = route_open.repeat(8, 1)
                # Note: shape is (batch, problem+1, problem+1)
            else:
                raise NotImplementedError
                
        self.route_open = route_open
        self.node_lateTW = node_lateTW
                
        # Const @Load_Problem
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        # Fill Reset_State
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_lateTW = node_lateTW
        self.reset_state.route_open = route_open
        
        # Fill Step_State
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        # Fill for masking use
        self.adj_matrices = adj_matrices
        # shape: (batch, problem+1, problem+1)

        if (node_lateTW.sum()>0.001):
            self.attribute_tw = True
        else:
            self.attribute_tw = False
        if (route_open.sum()>0.001):
            self.attribute_o = True
        else:
            self.attribute_o = False
        
        self.reset_state.attribute_tw = self.attribute_tw
        self.reset_state.attribute_o = self.attribute_o
        
        # For future multi-depot consideration:
        self.current_depot_xy = depot_xy.expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, 2)
    
    def reset(self):
        
        # Fill Dynamic-1
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        # Fill Dynamic-2
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.full(size=(self.batch_size, self.pomo_size), fill_value=self.vehicle_capacity, dtype=torch.float)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        # Fill for decoder's query vector use
        self.current_vehicle = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.collected_demand = torch.zeros(size=(self.batch_size, self.pomo_size))
        
        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        # Fill Step_State
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # Part of decoder query
        self.step_state.current_vehicle = self.current_vehicle
        self.step_state.current_time = self.current_time
        self.step_state.current_depot_xy = self.current_depot_xy
        self.step_state.collected_demand = self.collected_demand
        
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        """Environment step function, handles state transitions"""
        # selected.shape: (batch, pomo)
        
        # 1. Update basic state
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)

        # Check if depot is selected
        self.at_the_depot = (selected == 0)

        # 2. Calculate time consumption and update load
        if self.selected_node_list.size(2) > 1:  # Not the first step
            prev_node = self.selected_node_list[:, :, -2]  # Previous node
            prev_xy = self.depot_node_xy[self.BATCH_IDX, prev_node]
            current_xy = self.depot_node_xy[self.BATCH_IDX, selected]
            time_increment = torch.norm(current_xy - prev_xy, dim=2)
            self.load -= time_increment
            # shape: (batch, pomo)

            if (self.attribute_tw):
                self.current_time += time_increment
                # shape: (batch, pomo)
                self.current_time[self.at_the_depot] = 0  # refill time at the depot

            # 3. Update collected demand
            demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
            selected_demand = demand_list.gather(dim=2, index=selected[:, :, None]).squeeze(2)
            self.collected_demand += selected_demand 
    
            # 4. Update visited flags (special handling based on node type)
            self._update_visited_flags(selected)

            # 5. Update finished status
            # Check if all vehicles are used
            is_last_vehicle = (self.current_vehicle == self.num_vehicles)
            last_vehicle_finished = self.at_the_depot & is_last_vehicle
            self.finished = self.finished | last_vehicle_finished

            # All customer nodes are visited (only consider one-time visit nodes 48-97)
            one_time_nodes = torch.arange(self.original_node_count, self.problem_size + 1)
            all_one_time_visited = (self.visited_ninf_flag[:, :, one_time_nodes] != 0).all(dim=2)
            self.finished = self.finished | all_one_time_visited  # Maintain finished status

            # 6. Update vehicle switching logic, should be after is_last_vehicle check
            # When returning to depot and not the last vehicle, switch to next vehicle, and route not finished
            can_switch_vehicle = self.at_the_depot & (self.current_vehicle < self.num_vehicles)
            can_switch_vehicle = ~self.finished & can_switch_vehicle
            self.load[can_switch_vehicle] = self.vehicle_capacity  # Reset time limit
            self.current_vehicle[can_switch_vehicle] += 1
    
            # 7. Call independent method to update mask and finished status
            self.update_mask(selected = selected,
                            visited_ninf_flag = self.visited_ninf_flag,
                            current_load = self.load,
                            current_time = self.current_time,
                            finished = self.finished,
                            attribute_tw = self.attribute_tw,
                            attribute_o = self.attribute_o)

        # 8. Update step_state and return
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # Update part of decoder query
        self.step_state.current_vehicle = self.current_vehicle
        self.step_state.current_time = self.current_time
        self.step_state.current_depot_xy = self.current_depot_xy
        self.step_state.collected_demand = self.collected_demand

        done = self.finished.all()
        reward = self.collected_demand if done else None
        return self.step_state, reward, done
        
    def _update_visited_flags(self, selected):
        # Nodes 0 to original_node_count-1: reusable, including depot=0, including original_node_count-1, total 48
        # Nodes (original_node_count) to (problem_size): one-time visit, including original_node_count, including problem_size, total 50
        
        # For one-time visit nodes, mark as visited
        one_time_mask = selected >= self.original_node_count
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = torch.where(
            one_time_mask, float('-inf'), 0.0)
        
        # Special handling: when not at depot, mark depot as unvisited
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

    def update_mask(self, selected, visited_ninf_flag, current_load, current_time, finished, attribute_tw, attribute_o):
        
        # 0. Road connectivity mask (based on adjacency matrix)
        connectivity_mask = self.adj_matrices[self.BATCH_IDX, selected, :] != 1
        # shape: (batch_size, pomo_size, problem+1)

        # 1. Visited node mask (not selectable) (only applies to one-time visit nodes)
        visited_mask = visited_ninf_flag != 0  # shape: (batch, pomo, problem+1)

        # 2. Calculate time constraint mask
        current_xy = self.depot_node_xy[self.BATCH_IDX, selected, :]  # (batch,pomo,2)
        will_select_xy = self.depot_node_xy.unsqueeze(1).expand(-1, self.pomo_size, -1, -1)  # (batch,pomo,problem+1,2)
        current_xy_expanded = current_xy.unsqueeze(2)  # (batch,pomo,1,2)
        distance_to_target = torch.norm(current_xy_expanded - will_select_xy, dim=3)  # (batch,pomo,problem+1)
        
        # New: multiply distance by 2 for new nodes (48-97)
        # Create node type mask: True indicates new nodes (original_node_count to problem_size)
        node_indices = torch.arange(self.problem_size + 1).expand(self.batch_size, self.pomo_size, -1)
        is_new_node = node_indices >= self.original_node_count
        # shape: (batch, pomo, problem+1)
        
        # Multiply distance by 2 for new nodes
        distance_to_target = torch.where(is_new_node, 
                                       distance_to_target * 2, 
                                       distance_to_target)
        
        depot_xy = will_select_xy[:, :, 0:1, :]  # (batch,pomo,1,2)
        distance_back_to_depot = torch.norm(will_select_xy - depot_xy, dim=3)  # (batch,pomo,problem+1)

        if attribute_o:
            total_time_needed = distance_to_target
        else:
            total_time_needed = distance_to_target + distance_back_to_depot  # (batch,pomo,problem+1)
            
        time_exceeded = total_time_needed > current_load[:, :, None]  # (batch,pomo,problem+1)
        # (batch, pomo, problem_size+1)

        # 3. Original nodes (1 to original_node_count) cannot be visited three consecutive times in one route
        not_allow = torch.zeros_like(connectivity_mask)  # (batch, pomo, problem+1)
        
        if self.selected_node_list.size(2) > 1:  # At least one step has been selected
            # Get the last two selected nodes (shape: (batch, pomo, 2))
            recent_nodes = self.selected_node_list[:, :, -2:]  # (batch, pomo, 2)
            # When recent_nodes = self.selected_node_list[:, :, 1:], all nodes cannot be revisited
            # recent_nodes = self.selected_node_list[:, :, 1:]
            # Generate all non-depot node IDs (1 to problem_size) (shape: (problem_size,))
            node_ids = torch.arange(1, self.problem_size + 1, device=recent_nodes.device)  # (problem_size,)
            
            # Expand dimensions to support broadcast comparison: (batch, pomo, 2) -> (batch, pomo, 1, 2)
            recent_nodes_expanded = recent_nodes.unsqueeze(2)  # (batch, pomo, 1, 2)
            
            # Expand node ID dimensions: (problem_size,) -> (1, 1, problem_size, 1)
            node_ids_expanded = node_ids.view(1, 1, -1, 1)  # (1, 1, problem_size, 1)
            
            # Element-wise comparison: check if any node appears in the last two selections (shape: (batch, pomo, problem_size, 2))
            node_counts = (recent_nodes_expanded == node_ids_expanded).sum(dim=3)  # (batch, pomo, problem_size, 2)
            
            # Generate prohibition mask: if occurrence count ≥ 1, prohibit selecting that node (shape: (batch, pomo, problem_size))
            node_not_allow = (node_counts >= 1)  # (batch, pomo, problem_size)
            
            # Write prohibition mask to total mask (note node ID starts from 1, corresponding to indices 1 to problem_size)
            not_allow[:, :, 1:self.problem_size + 1] = node_not_allow  # (batch, pomo, problem+1)


        # Supplementary mask 123: Visit time cannot exceed the shortest time of the node
        if attribute_tw:
            current_xy = self.depot_node_xy[self.BATCH_IDX, selected, :]  # (batch,pomo,2)
            will_select_xy = self.depot_node_xy.unsqueeze(1).expand(-1, self.pomo_size, -1, -1)  # (batch,pomo,problem+1,2)
            current_xy_expanded = current_xy.unsqueeze(2)  # (batch,pomo,1,2)
            distance_to_target = torch.norm(current_xy_expanded - will_select_xy, dim=3)  # (batch,pomo,problem+1)
            time_too_late = current_time[:,:,None] + distance_to_target[:,:,1:] > self.node_lateTW[:, None, :].expand(-1, self.pomo_size, -1)
            # (batch,pomo,problem)
            time_too_late = torch.cat((torch.zeros(size=(self.batch_size, self.pomo_size, 1), dtype=torch.bool), 
                                         time_too_late), dim=2)
            # (batch,pomo,1+problem)
            total_mask = connectivity_mask | visited_mask | time_exceeded | not_allow | time_too_late
        else:
            # 4. Merge constraints to generate total mask (logical OR)
            # As long as any condition of "connection matrix" or "visited" or "insufficient load" or "consecutive visits" is met, the node is not selectable
            total_mask = connectivity_mask | visited_mask | time_exceeded | not_allow
            # shape: (batch, pomo, problem+1)
        
        # 5. Generate numerical mask (-inf indicates not selectable)
        self.ninf_mask = torch.where(total_mask,          # Condition: nodes to be masked
                                    float('-inf'),        # Non-selectable nodes assigned -inf
                                    0.0)                  # Selectable nodes remain 0
        # shape: (batch, pomo, problem+1)

        # 6. Allow depot visit after task completion
        self.ninf_mask[:, :, 0][finished] = 0  # Depot node selectable for completed episodes
        self.ninf_mask[:, :, 1:][finished] = float('-inf')  # Only depot node selectable for completed episodes
        # shape: (batch, pomo, problem+1)

        # 7. Special handling: when all customer nodes are not selectable, allow depot visit (avoid deadlock)
        customer_mask = self.ninf_mask[:, :, 1:] 
        all_customer_inf = (customer_mask != 0).all(dim=2)
        self.ninf_mask[:, :, 0][all_customer_inf] = 0 
        
    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances