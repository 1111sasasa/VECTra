import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv
from basemodel import BaseModel
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import math


class JGRMModel(BaseModel):
    def __init__(self, vocab_size, route_max_len, road_feat_num, road_embed_size, gps_feat_num, gps_embed_size,
                 route_embed_size, hidden_size, edge_index, drop_edge_rate, drop_route_rate, drop_road_rate,
                 mode='p', add_temporal_bias=True, temporal_bias_dim=64):
        super(JGRMModel, self).__init__()

        self.vocab_size = vocab_size  # 路段数量
        self.edge_index = torch.tensor(edge_index).cuda()  # 将输入的边索引转换为 GPU 上的 PyTorch 张量
        self.mode = mode
        self.drop_edge_rate = drop_edge_rate
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim

        # node embedding
        self.route_padding_vec = torch.zeros(1, road_embed_size, requires_grad=True).cuda()#（1，128）的全0向量，用来填充[[0，0，0……0]]
        self.node_embedding = nn.Embedding(vocab_size, road_embed_size)
        self.node_embedding.requires_grad_(True)

        # time embedding
        self.minute_embedding = nn.Embedding(1440 + 1, route_embed_size)  # 0 is mask
        self.week_embedding = nn.Embedding(7 + 1, route_embed_size)  # 0 is mask
        self.delta_embedding = IntervalEmbedding(100, route_embed_size)  # -1 is mask

        # route encoding
        self.graph_encoder = GraphEncoder(road_embed_size, route_embed_size)  # 聚合邻居信息
        self.position_embedding1 = nn.Embedding(route_max_len, route_embed_size)  # 对路段顺序编码
        self.fc1 = nn.Linear(route_embed_size, hidden_size)  # route fuse time ffn
        self.route_encoder = TransformerModel(hidden_size, 8, hidden_size, 4, drop_route_rate,
                                              add_temporal_bias=add_temporal_bias,
                                              temporal_bias_dim=temporal_bias_dim)

        # gps encoding
        self.gps_linear = nn.Linear(gps_feat_num, gps_embed_size)
        self.gps_intra_encoder = nn.GRU(gps_embed_size, gps_embed_size, bidirectional=True, batch_first=True)
        self.gps_inter_encoder = nn.GRU(gps_embed_size, gps_embed_size, bidirectional=True, batch_first=True)

        # cl project head
        self.gps_proj_head = nn.Linear(2 * gps_embed_size, hidden_size)
        self.route_proj_head = nn.Linear(hidden_size, hidden_size)

        # shared transformer
        self.position_embedding2 = nn.Embedding(route_max_len, hidden_size)
        self.modal_embedding = nn.Embedding(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sharedtransformer = TransformerModel(hidden_size, 4, hidden_size, 2, drop_road_rate,
                                                  add_temporal_bias=False)

        # mlm classifier head
        self.gps_mlm_head = nn.Linear(hidden_size, vocab_size)
        self.route_mlm_head = nn.Linear(hidden_size, vocab_size)

        # matching
        self.matching_predictor = nn.Linear(hidden_size * 2, 2)
        self.register_buffer("gps_queue", torch.randn(hidden_size, 2048))
        self.register_buffer("route_queue", torch.randn(hidden_size, 2048))

        self.image_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.route_queue, dim=0)

        # Initialize temporal bias parameters if needed
        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def compute_temporal_bias(self, temporal_mat):
        temporal_mat = 1.0 / torch.log(torch.exp(torch.tensor(1.0).cuda()) + temporal_mat)
        if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
            temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                self.temporal_mat_bias_1(temporal_mat.unsqueeze(-1)),
                negative_slope=0.2)).squeeze(-1)
        elif self.temporal_bias_dim == -1:
            temporal_mat = temporal_mat * self.temporal_mat_bias.expand(temporal_mat.size())

        return temporal_mat

    def encode_graph(self, drop_rate=0.2):
        node_emb = self.node_embedding.weight
        edge_index = dropout_adj(self.edge_index, p=drop_rate)[0]
        node_enc = self.graph_encoder(node_emb, edge_index)
        return node_enc

    def encode_route(self, route_data, route_assign_mat, masked_route_assign_mat):
        if self.mode == 'p':
            lookup_table = torch.cat([self.node_embedding.weight, self.route_padding_vec], 0)
        else:
            node_enc = self.encode_graph(self.drop_edge_rate)
            lookup_table = torch.cat([node_enc, self.route_padding_vec], 0)

        # 先对原始序列进行mask，然后再进行序列建模，防止信息泄露
        batch_size, max_seq_len = masked_route_assign_mat.size()

        src_key_padding_mask = (route_assign_mat == self.vocab_size)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 为padding位

        route_emb = torch.index_select(
                lookup_table, 0, masked_route_assign_mat.int().view(-1)).view(batch_size, max_seq_len, -1)

        #Compute temporal bias matrix
        # Compute temporal bias matrix using actual timestamps
        temporal_mat = None
        if self.add_temporal_bias and route_data is not None:
            # 第4个维度存储的是实际时间戳
            time_stamps = route_data[:, :, 3].float()  # (B, T) 实际时间戳

            # 计算时间差矩阵
            time_diff = time_stamps.unsqueeze(2) - time_stamps.unsqueeze(1)  # (B, T, T) 时间差矩阵

            # 处理填充值
            padding_mask = (route_assign_mat == self.vocab_size)  # (B, T) 填充位置掩码
            padding_mask_matrix = padding_mask.unsqueeze(2) | padding_mask.unsqueeze(1)  # (B, T, T)
            time_diff = time_diff.masked_fill(padding_mask_matrix, 0)  # 将填充位置的时间差设为0

            # 计算时间偏置
            temporal_mat = self.compute_temporal_bias(time_diff.abs())
        # time embedding
        if route_data is None:
            week_emb = self.week_embedding.weight.detach()[1:].mean(dim=0)
            min_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
            delta_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
        else:
            week_data = route_data[:, :, 0].long()
            min_data = route_data[:, :, 1].long()
            delta_data = route_data[:, :, 2].float()
            week_emb = self.week_embedding(week_data)
            min_emb = self.minute_embedding(min_data)
            delta_emb = self.delta_embedding(delta_data)

        # position embedding
        position = torch.arange(route_emb.shape[1]).long().cuda()
        pos_emb = position.unsqueeze(0).repeat(route_emb.shape[0], 1)
        pos_emb = self.position_embedding1(pos_emb)

        # fuse info
        route_emb = route_emb + pos_emb + week_emb + min_emb + delta_emb
        route_emb = self.fc1(route_emb)

        # Pass temporal_mat to transformer
        route_enc = self.route_encoder(route_emb, None, src_key_padding_mask, temporal_mat=temporal_mat)
        route_enc = torch.where(torch.isnan(route_enc), torch.full_like(route_enc, 0), route_enc)

        route_unpooled = route_enc * pool_mask.repeat(1, 1, route_enc.shape[-1])
        route_pooled = route_unpooled.sum(1) / pool_mask.sum(1).clamp(min=1)

        return route_unpooled, route_pooled

    def encode_gps(self, gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length):
        # gps_data 先输入 gps_encoder, 输出每个step的output，选择路段对应位置的gps点的output进行pooling作为路段的表示
        gps_data = self.gps_linear(gps_data)#变换后的 gps_data 形状为 (batch_size, gps_max_length, gps_embed_size)。

        # mask features  对 GPS 数据进行掩码（Masking）处理，确保模型在计算时忽略填充部分（Padding）和被掩码的部分（Masked Tokens）。
        gps_src_key_padding_mask = (masked_gps_assign_mat == self.vocab_size)   #生成的布尔矩阵
        gps_mask_mat = (1 - gps_src_key_padding_mask.int()).unsqueeze(-1).repeat(1, 1, gps_data.shape[-1]) # 0 为padding位
        masked_gps_data = gps_data * gps_mask_mat # (batch_size,gps_max_len,feat_num)

        # flatten gps data 便于进行路段内gru的并行
        flattened_gps_data, route_length = self.gps_flatten(masked_gps_data, gps_length) # flattened_gps_data (road_num, max_pt_len ,gps_fea_size)
        _, gps_emb = self.gps_intra_encoder(flattened_gps_data) # gps_emb (1, road_num, gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state
        gps_emb = gps_emb[-1] # 只保留前向的表示
        # gps_emb = torch.cat([gps_emb[0].squeeze(0), gps_emb[1].squeeze(0)],dim=-1) # 前后向表示拼接

        # stack gps emb 便于进行路段间gru的计算
        stacked_gps_emb = self.route_stack(gps_emb, route_length) # stacked_gps_emb (batch_size, max_route_len, gps_embed_size)
        gps_emb, _ = self.gps_inter_encoder(stacked_gps_emb)  # (batch_size, max_route_len, 2*gps_embed_size) # 不输入hidden默认输入全0为序列的hidden state

        route_src_key_padding_mask = (masked_route_assign_mat == self.vocab_size).transpose(0, 1)
        route_pool_mask = (1 - route_src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1) # 包含mask的长度
        # 对于单路段mask，可能存在整个route都被mask掉的情况，此时pool_mask.sum(1)中有0值，令其最小值为1防止0除
        gps_pooled = gps_emb.sum(1) / route_pool_mask.sum(1).clamp(min=1) # mask 后的有值的路段数量，比路段长度要短
        gps_unpooled = gps_emb

        return gps_unpooled, gps_pooled

    def route_stack(self, gps_emb, route_length):   #将分散的 gps_emb（按路段平铺）重新按原始轨迹分组，生成一个三维张量。
        # flatten_gps_data tensor = (real_len, max_gps_in_route_len, emb_size)
        # route_length dict = { key:tid, value: road_len }
        values = list(route_length.values())
        route_max_len = max(values)
        data_list = []
        for idx in range(len(route_length)):
            start_idx = sum(values[:idx])
            end_idx = sum(values[:idx+1])
            data = gps_emb[start_idx:end_idx]
            data_list.append(data)

        stacked_gps_emb = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)

        return stacked_gps_emb

    def gps_flatten(self, gps_data, gps_length):
        # 把gps_data按照gps_assign_mat做形变，把每个路段上的gps点单独拿出来，拼成一个新的tensor (road_num, gps_max_len, gps_feat_num)，
        # 该tensor用于输入GRU进行并行计算
        traj_num, gps_max_len, gps_feat_num = gps_data.shape
        flattened_gps_list = []
        route_index = {}  #键是第几条轨迹，值是该轨迹中有多少路段
        for idx in range(traj_num):
            gps_feat = gps_data[idx] # (max_len, feat_num)   取出当前这条轨迹中所有的gps点
            length_list = gps_length[idx] # (max_len, 1) [7,9,12,1,0,0,0,0,0,0] # padding_value = 0
            # 遍历每个轨迹中的路段
            for _idx, length in enumerate(length_list):
                if length != 0:   #跳过无效的填充路段（0）
                    start_idx = sum(length_list[:_idx])
                    end_idx = start_idx + length_list[_idx]
                    cnt = route_index.get(idx, 0)
                    route_index[idx] = cnt+1
                    road_feat = gps_feat[start_idx:end_idx]  #当前路段的GPS数据块（road_feat）,从当前轨迹的GPS数据中，切片提取属于当前路段的GPS点
                    flattened_gps_list.append(road_feat)

        flattened_gps_data = rnn_utils.pad_sequence(flattened_gps_list, padding_value=0, batch_first=True) # (road_num, gps_max_len, gps_feat_num)
        #flattened_gps_data形状三维(所有路段总数, max_pts_per_road, gps_feat_num)

        return flattened_gps_data, route_index

    def encode_joint(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat):
        max_len = torch.max((route_assign_mat != self.vocab_size).int().sum(1)).item()
        max_len = max_len * 2 + 2
        data_list = []
        mask_list = []
        route_length = [length[length != self.vocab_size].shape[0] for length in route_assign_mat]

        modal_emb0 = self.modal_embedding(torch.tensor(0).cuda())
        modal_emb1 = self.modal_embedding(torch.tensor(1).cuda())

        for i, length in enumerate(route_length):
            route_road_token = route_road_rep[i][:length]
            gps_road_token = gps_road_rep[i][:length]
            route_cls_token = route_traj_rep[i].unsqueeze(0)
            gps_cls_token = gps_traj_rep[i].unsqueeze(0)

            position = torch.arange(length + 1).long().cuda()
            pos_emb = self.position_embedding2(position)

            route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
            modal_emb = modal_emb0.unsqueeze(0).repeat(length + 1, 1)
            route_emb = route_emb + pos_emb + modal_emb
            route_emb = self.fc2(route_emb)

            gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
            modal_emb = modal_emb1.unsqueeze(0).repeat(length + 1, 1)
            gps_emb = gps_emb + pos_emb + modal_emb
            gps_emb = self.fc2(gps_emb)

            data = torch.cat([gps_emb, route_emb], dim=0)
            data_list.append(data)

            mask = torch.tensor([False] * data.shape[0]).cuda()
            mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        # Pass temporal_mat to transformer
        joint_emb = self.sharedtransformer(joint_data, None, mask_mat)

        # 每一行的0 和 length+1 对应的是 gps_traj_rep 和 route_traj_rep
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack([joint_emb[i, length + 1] for i, length in enumerate(route_length)], dim=0)

        gps_road_rep = rnn_utils.pad_sequence([joint_emb[i, 1:length + 1] for i, length in enumerate(route_length)],
                                              padding_value=0, batch_first=True)
        route_road_rep = rnn_utils.pad_sequence(
            [joint_emb[i, length + 2:2 * length + 2] for i, length in enumerate(route_length)],
            padding_value=0, batch_first=True)

        return gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep

    def forward(self, route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat,
                gps_length):
        gps_road_rep, gps_traj_rep = self.encode_gps(gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length)
        route_road_rep, route_traj_rep = self.encode_route(route_data, route_assign_mat, masked_route_assign_mat)
        gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep = self.encode_joint(
            route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat)

        return gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, \
            gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep

# GAT
class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphEncoder, self).__init__()
        # update road edge features using GAT
        self.layer1 = GATConv(input_size, output_size)
        self.layer2 = GATConv(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.layer1(x, edge_index))
        x = self.activation(self.layer2(x, edge_index))
        return x



class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3,
                 add_temporal_bias=False, temporal_bias_dim=64):
        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout,
                                                 add_temporal_bias=add_temporal_bias,
                                                 temporal_bias_dim=temporal_bias_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        encoder_layer = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers)
        self.add_temporal_bias = add_temporal_bias

    def forward(self, src, src_mask, src_key_padding_mask, temporal_mat=None):
        if self.add_temporal_bias and temporal_mat is not None:
            output = src
            temporal_mat = temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            for layer in self.transformer_encoder.layers:
                output, _ = layer(output, src_mask, src_key_padding_mask, temporal_mat=temporal_mat)
        else:
            output = self.transformer_encoder2(src, src_mask, src_key_padding_mask)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 add_temporal_bias=False, temporal_bias_dim=64):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttentionWithTemporalBias(
            d_model, nhead, dropout=dropout,
            add_temporal_bias=add_temporal_bias,
            temporal_bias_dim=temporal_bias_dim)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, temporal_mat=None):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            temporal_mat=temporal_mat)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class MultiheadAttentionWithTemporalBias(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 add_temporal_bias=False, temporal_bias_dim=64):
        super(MultiheadAttentionWithTemporalBias, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.add_temporal_bias = add_temporal_bias

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if self.add_temporal_bias:
            if temporal_bias_dim != 0 and temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(temporal_bias_dim, 1, bias=True)
            elif temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, temporal_mat=None):
    #query\key\value shape: torch.Size([64, 46, 256])
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        #标准qkv计算
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        #多头拆分
        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)
        #注意力分数计算
        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        # Add temporal bias if provided
        if self.add_temporal_bias and temporal_mat is not None:
            if hasattr(self, 'temporal_mat_bias_1'):
                temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                    self.temporal_mat_bias_1(temporal_mat.unsqueeze(-1)),
                    negative_slope=0.2)).squeeze(-1)
            elif hasattr(self, 'temporal_mat_bias'):
                temporal_mat = temporal_mat * self.temporal_mat_bias.expand(temporal_mat.size())

            temporal_mat = temporal_mat.expand(-1, self.num_heads, -1, -1)
            temporal_mat = temporal_mat.reshape(bsz * self.num_heads, tgt_len, src_len)

            attn_output_weights += temporal_mat

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        #后续标准处理
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_output_weights

class IntervalEmbedding(nn.Module):
    def __init__(self, num_bins, hidden_size):
        super(IntervalEmbedding, self).__init__()
        self.layer1 = nn.Linear(1, num_bins)
        self.emb = nn.Embedding(num_bins, hidden_size)
        self.activation = nn.Softmax()

    def forward(self, x):
        logit = self.activation(self.layer1(x.unsqueeze(-1)))
        output = logit @ self.emb.weight
        return output
