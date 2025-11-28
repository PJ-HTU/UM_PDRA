import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PDRAModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = PDRA_Encoder(**model_params)
        self.decoder = PDRA_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state, vehicle_config=None):

        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_lateTW = reset_state.node_lateTW
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        node_xy_demand_TW = torch.cat((node_xy_demand, node_lateTW[:, :, None]),dim=2)
        # shape: (batch, problem, 4)

        route_open = reset_state.route_open
        # shape: (batch, 1)
        
        batch_size = depot_xy.size(0)
        device = depot_xy.device                             
        config_tensor = torch.tensor([vehicle_config['num_vehicles'], vehicle_config['vehicle_capacity']],
                        dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # shape: (batch, 1, 2)
        depot_xy_with_config = torch.cat((depot_xy, config_tensor, route_open[:,None,:]), dim=2)  # shape: (batch, 1, 5)

        self.encoded_nodes = self.encoder(depot_xy_with_config, node_xy_demand_TW)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:    # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
           
            probs = self.decoder(encoded_last_node, state.load, state.current_time, 
                                 state.current_vehicle, state.current_depot_xy, ninf_mask = state.ninf_mask)
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes

########################################
# ENCODER
########################################

class PDRA_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # Depot input dimension 5: coordinates + num_vehicles + capacity + is_open
        self.embedding_depot = nn.Linear(5, embedding_dim)
        
        # Node input dimension 4: coordinates + information_value + acceptable_latest_arrival_time
        self.embedding_node = nn.Linear(4, embedding_dim)
        
        self.layers = nn.ModuleList([RouteFinderEncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy_with_config, node_xy_demand):
        embedded_depot = self.embedding_depot(depot_xy_with_config)  # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)          # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)     # shape: (batch, problem+1, embedding)
        for layer in self.layers:
            out = layer(out)

        return out  # shape: (batch, problem+1, embedding)


class RouteFinderEncoderLayer(nn.Module):
    """RouteLink Transformer Encoder Layer with RMS Norm, SwigLU, and Flash Attention"""
    
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Multi-Head Attention components
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # RMS Normalization layers (pre-norm architecture)
        self.rms_norm_1 = RMSNorm(embedding_dim)
        self.rms_norm_2 = RMSNorm(embedding_dim)
        
        # SwigLU Feed Forward Network
        self.feed_forward = SwigLUFeedForward(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        # Pre-norm Multi-Head Attention
        normalized_input = self.rms_norm_1(input1)
        
        q = reshape_by_heads(self.Wq(normalized_input), head_num=head_num)
        k = reshape_by_heads(self.Wk(normalized_input), head_num=head_num)
        v = reshape_by_heads(self.Wv(normalized_input), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        # Flash Attention
        out_concat = flash_multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        # First residual connection
        out1 = input1 + multi_head_out

        # Pre-norm Feed Forward
        normalized_out1 = self.rms_norm_2(out1)
        ff_out = self.feed_forward(normalized_out1)
        
        # Second residual connection
        out2 = out1 + ff_out

        return out2
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class PDRA_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Query encoding: last visited node encoding h(embedding_dim) + current remaining vehicle time + current time window + current vehicle count + initial depot coordinates
        self.Wq_last = nn.Linear(embedding_dim + 1 + 1 + 1 + 2, head_num * qkv_dim, bias=False)  

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, encoded_last_node, load, current_time, current_vehicle, current_depot_xy, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # current_time.shape: (batch, pomo)
        # current_vehicle.shape: (batch, pomo)
        # current_depot_xy.shape: (batch, pomo, 2)
        # ninf_mask.shape: (batch, pomo, problem+1)

        head_num = self.model_params['head_num']
        batch_size = encoded_last_node.size(0)
        pomo_size = encoded_last_node.size(1)

        # Query encoding: last visited node encoding h(embedding_dim) + current remaining vehicle time + current time window + current vehicle count + initial depot coordinates
        # embedding_dim + 1 + 1 + 1 + 2
        
        # Concatenate input features
        input_cat = torch.cat((encoded_last_node, load[:, :, None], current_time[:, :, None],
                               current_vehicle[:, :, None], current_depot_xy), dim=2)
        # shape = (batch, pomo, EMBEDDING_DIM+1+2)

        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # Use Flash Attention for decoder as well
        out_concat = flash_multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        # Single-Head Attention, for probability calculation
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NEW COMPONENTS: RMS NORM, SwigLU, FLASH ATTENTION
########################################

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x.shape: (batch, seq_len, dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwigLUFeedForward(nn.Module):
    """SwigLU Activation Function Feed Forward Network"""
    
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        
        # SwigLU requires two linear projections for gating
        self.gate_proj = nn.Linear(embedding_dim, ff_hidden_dim, bias=False)
        self.up_proj = nn.Linear(embedding_dim, ff_hidden_dim, bias=False)
        self.down_proj = nn.Linear(ff_hidden_dim, embedding_dim, bias=False)

    def forward(self, x):
        # x.shape: (batch, seq_len, embedding_dim)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwigLU: SwiGLU(x, W, V, b, c) = (Swish(xW + b) ⊙ (xV + c))
        # Swish(x) = x * sigmoid(x)
        swish_gate = gate * torch.sigmoid(gate)
        gated = swish_gate * up
        
        output = self.down_proj(gated)
        return output


def flash_multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """
    Flash Attention implementation using PyTorch's optimized scaled_dot_product_attention
    when available, otherwise falls back to efficient manual implementation
    """
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    # Prepare attention mask if needed
    attn_mask = None
    if rank2_ninf_mask is not None:
        attn_mask = rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        if attn_mask is None:
            attn_mask = rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
        else:
            attn_mask = attn_mask + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    # Try to use PyTorch's optimized Flash Attention if available
    try:
        if hasattr(F, 'scaled_dot_product_attention') and attn_mask is None:
            # PyTorch 2.0+ optimized attention (includes Flash Attention optimizations)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        else:
            # Manual implementation with memory-efficient attention
            out = _efficient_attention(q, k, v, attn_mask)
    except:
        # Fallback to manual implementation
        out = _efficient_attention(q, k, v, attn_mask)
    
    # Reshape output
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


def _efficient_attention(q, k, v, attn_mask=None):
    """Memory-efficient attention computation"""
    batch_s, head_num, n, key_dim = q.shape
    
    # Compute attention scores
    scale_factor = 1.0 / math.sqrt(key_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
    
    # Apply mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask
    
    # Softmax attention weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    out = torch.matmul(attn_weights, v)
    
    return out


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """Legacy multi-head attention (kept for compatibility)"""
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


# Legacy normalization classes (kept for backward compatibility)
class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans


class FeedForward(nn.Module):
    """Legacy Feed Forward (kept for backward compatibility)"""
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))