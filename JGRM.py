import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv
from basemodel import BaseModel
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import math
import torch.utils.checkpoint as checkpoint
from vision_encoder import TimeVLMVisionEncoder


class JGRMModel(BaseModel):
    def __init__(self, vocab_size, route_max_len, road_feat_num, road_embed_size, gps_feat_num, gps_embed_size,
                 route_embed_size, hidden_size, edge_index, drop_edge_rate, drop_route_rate, drop_road_rate,
                 mode='p', add_temporal_bias=True, temporal_bias_dim=64, use_vision=False, vision_image_size=224,
                 vision_periodicity=24, vision_hidden_dim=64, vision_output_channels=3,
                 clip_model_name='ViT-B-32', clip_pretrained='openai', freeze_clip=False,
                 vision_feature_idx=(1, 2, 3, 4, 5, 6, 7), use_vision_gate=False,
                 freeze_ts_to_image=False, use_checkpoint=False, use_vision_in_joint=True,
                 gps_intra_chunk_size=None, vision_fuse_after_gru=False, vision_fuse_after_joint=False,
                 use_route_vision=False, route_vision_feature_idx=(0, 1, 2),
                 route_vision_stats=None, route_vision_use_log1p=False,
                 use_vision_pair_fuse=False, use_vision_pair_gate=True,
                 fusion_type='shared', use_modality_embedding=True, cross_modal_num_heads=4,
                 cross_modal_num_layers=1):
        super(JGRMModel, self).__init__()

        self.vocab_size = vocab_size  # 路段数量
        self.edge_index = torch.tensor(edge_index).cuda()  # 将输入的边索引转换为 GPU 上的 PyTorch 张量
        self.mode = mode
        self.drop_edge_rate = drop_edge_rate
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_vision = use_vision
        self.vision_feature_idx = vision_feature_idx
        self.use_vision_gate = use_vision_gate
        self.use_checkpoint = use_checkpoint
        self.use_vision_in_joint = use_vision_in_joint
        self.gps_intra_chunk_size = gps_intra_chunk_size
        self.vision_fuse_after_gru = vision_fuse_after_gru
        self.vision_fuse_after_joint = vision_fuse_after_joint
        self.use_route_vision = use_route_vision
        self.route_vision_feature_idx = route_vision_feature_idx
        self.route_vision_stats = route_vision_stats
        self.route_vision_use_log1p = route_vision_use_log1p
        self.use_vision_pair_fuse = use_vision_pair_fuse
        self.use_vision_pair_gate = use_vision_pair_gate
        self.fusion_type = fusion_type
        self.use_modality_embedding = use_modality_embedding

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
        modal_count = 3 if (self.use_vision and self.use_vision_in_joint) else 2
        self.modal_embedding = nn.Embedding(modal_count, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sharedtransformer = TransformerModel(hidden_size, 4, hidden_size, 2, drop_road_rate,
                                                  add_temporal_bias=False)
        self.cross_modal_fusion = BiCrossModalFusion(
            hidden_size=hidden_size,
            num_heads=cross_modal_num_heads,
            dropout=drop_road_rate,
            num_layers=cross_modal_num_layers,
        )

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

        # vision encoder (optional)
        if self.use_vision:
            self.vision_encoder = TimeVLMVisionEncoder(
                input_dim=len(self.vision_feature_idx),
                image_size=vision_image_size,
                periodicity=vision_periodicity,
                hidden_dim=vision_hidden_dim,
                output_channels=vision_output_channels,
                clip_model_name=clip_model_name,
                clip_pretrained=clip_pretrained,
                freeze_clip=freeze_clip,
                freeze_ts_to_image=freeze_ts_to_image,
            )
            self.vision_proj_head = nn.Linear(self.vision_encoder.output_dim, hidden_size)
            if self.use_route_vision:
                self.route_vision_encoder = TimeVLMVisionEncoder(
                    input_dim=len(self.route_vision_feature_idx),
                    image_size=vision_image_size,
                    periodicity=vision_periodicity,
                    hidden_dim=vision_hidden_dim,
                    output_channels=vision_output_channels,
                    clip_model_name=clip_model_name,
                    clip_pretrained=clip_pretrained,
                    freeze_clip=freeze_clip,
                    freeze_ts_to_image=freeze_ts_to_image,
                )
                self.route_vision_proj_head = nn.Linear(self.route_vision_encoder.output_dim, hidden_size)
                self.vision_fuse_route = nn.Linear(hidden_size * 2, hidden_size)
            if self.use_vision_pair_fuse:
                self.gps_vision_fuse = nn.Linear(hidden_size * 2, hidden_size)
                self.route_vision_fuse = nn.Linear(hidden_size * 2, hidden_size)
                if self.use_vision_pair_gate:
                    self.gps_vision_gate = nn.Sequential(
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    )
                    self.route_vision_gate = nn.Sequential(
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1),
                    )
            if self.use_vision_gate:
                self.gps_gate_proj = nn.Linear(2 * gps_embed_size, hidden_size)
                self.vision_gate = nn.Sequential(
                    nn.Linear(hidden_size * 3, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1),
                )
                self.vision_fuse = nn.Linear(hidden_size * 2, hidden_size)
            if self.vision_fuse_after_gru:
                self.vision_fuse_after_gru_route = nn.Linear(hidden_size * 2, hidden_size)
                self.vision_fuse_after_gru_gps = nn.Linear(hidden_size * 2, hidden_size)
            if self.vision_fuse_after_joint:
                self.vision_fuse_after_joint_route = nn.Linear(hidden_size * 2, hidden_size)
                self.vision_fuse_after_joint_gps = nn.Linear(hidden_size * 2, hidden_size)

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
        if getattr(self, 'add_temporal_bias', False) and route_data is not None:
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
        flattened_gps_list, route_length = self.gps_flatten(masked_gps_data, gps_length)
        gps_emb_chunks = []
        chunk_size = self.gps_intra_chunk_size or len(flattened_gps_list)
        for start in range(0, len(flattened_gps_list), chunk_size):
            chunk_list = flattened_gps_list[start:start + chunk_size]
            chunk_data = rnn_utils.pad_sequence(chunk_list, padding_value=0, batch_first=True)
            _, chunk_emb = self.gps_intra_encoder(chunk_data)
            gps_emb_chunks.append(chunk_emb[-1])
        gps_emb = torch.cat(gps_emb_chunks, dim=0)
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
        route_index = {}
        for idx in range(traj_num):
            gps_feat = gps_data[idx]
            length_list = gps_length[idx]
            for _idx, length in enumerate(length_list):
                if length != 0:
                    start_idx = sum(length_list[:_idx])
                    end_idx = start_idx + length_list[_idx]
                    cnt = route_index.get(idx, 0)
                    route_index[idx] = cnt + 1
                    road_feat = gps_feat[start_idx:end_idx]
                    flattened_gps_list.append(road_feat)

        return flattened_gps_list, route_index

    def encode_joint(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat, vision_traj_rep=None):
        max_len = torch.max((route_assign_mat != self.vocab_size).int().sum(1)).item()
        max_len = max_len * 2 + 2
        data_list = []
        mask_list = []
        route_length = [length[length != self.vocab_size].shape[0] for length in route_assign_mat]

        modal_emb0 = self.modal_embedding(torch.tensor(0).cuda()) if self.use_modality_embedding else 0.0
        modal_emb1 = self.modal_embedding(torch.tensor(1).cuda()) if self.use_modality_embedding else 0.0
        modal_emb2 = None

        if self.use_vision_in_joint and vision_traj_rep is not None and self.use_modality_embedding:
            modal_emb2 = self.modal_embedding(torch.tensor(2).cuda())

        for i, length in enumerate(route_length):
            route_road_token = route_road_rep[i][:length]
            gps_road_token = gps_road_rep[i][:length]
            route_cls_token = route_traj_rep[i].unsqueeze(0)
            gps_cls_token = gps_traj_rep[i].unsqueeze(0)

            position = torch.arange(length + 1).long().cuda()
            pos_emb = self.position_embedding2(position)

            route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
            if self.use_modality_embedding:
                modal_emb = modal_emb0.unsqueeze(0).repeat(length + 1, 1)
                route_emb = route_emb + pos_emb + modal_emb
            else:
                route_emb = route_emb + pos_emb
            route_emb = self.fc2(route_emb)

            gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
            if self.use_modality_embedding:
                modal_emb = modal_emb1.unsqueeze(0).repeat(length + 1, 1)
                gps_emb = gps_emb + pos_emb + modal_emb
            else:
                gps_emb = gps_emb + pos_emb
            gps_emb = self.fc2(gps_emb)

            if self.fusion_type == 'cross_modal':
                gps_mask = torch.tensor([False] * gps_emb.shape[0], device=gps_emb.device)
                route_mask = torch.tensor([False] * route_emb.shape[0], device=route_emb.device)
                gps_out, route_out = self.cross_modal_fusion(
                    gps_emb.unsqueeze(0), route_emb.unsqueeze(0),
                    gps_key_padding_mask=gps_mask.unsqueeze(0),
                    route_key_padding_mask=route_mask.unsqueeze(0),
                )
                gps_out = gps_out.squeeze(0)
                route_out = route_out.squeeze(0)
                if self.use_vision_in_joint and vision_traj_rep is not None:
                    vision_cls_token = vision_traj_rep[i].unsqueeze(0)
                    vision_pos_emb = self.position_embedding2(torch.tensor([0]).cuda())
                    if self.use_modality_embedding:
                        vision_emb = vision_cls_token + vision_pos_emb + modal_emb2.unsqueeze(0)
                    else:
                        vision_emb = vision_cls_token + vision_pos_emb
                    vision_emb = self.fc2(vision_emb)
                    data = torch.cat([gps_out, route_out, vision_emb], dim=0)
                else:
                    data = torch.cat([gps_out, route_out], dim=0)
            else:
                data = torch.cat([gps_emb, route_emb], dim=0)

                if self.use_vision_in_joint and vision_traj_rep is not None:
                    vision_cls_token = vision_traj_rep[i].unsqueeze(0)
                    vision_pos_emb = self.position_embedding2(torch.tensor([0]).cuda())
                    if self.use_modality_embedding:
                        vision_emb = vision_cls_token + vision_pos_emb + modal_emb2.unsqueeze(0)
                    else:
                        vision_emb = vision_cls_token + vision_pos_emb
                    vision_emb = self.fc2(vision_emb)
                    data = torch.cat([data, vision_emb], dim=0)

            data_list.append(data)

            mask = torch.tensor([False] * data.shape[0]).cuda()
            mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        # Pass temporal_mat to transformer
        if self.fusion_type == 'cross_modal':
            joint_emb = joint_data
        else:
            if self.use_checkpoint and self.training:
                def _shared_forward(x, mask):
                    return self.sharedtransformer(x, None, mask)

                joint_emb = checkpoint.checkpoint(_shared_forward, joint_data, mask_mat)
            else:
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

    def encode_vision(self, gps_data, gps_assign_mat=None):
        if not self.use_vision:
            return None
        feature_idx = list(self.vision_feature_idx)
        x_enc = gps_data[:, :, feature_idx]
        if gps_assign_mat is not None:
            padding_mask = (gps_assign_mat == self.vocab_size).unsqueeze(-1)
            x_enc = x_enc.masked_fill(padding_mask, 0.0)
        vision_traj_rep = self.vision_encoder(x_enc)
        vision_traj_rep = self.vision_proj_head(vision_traj_rep)
        return vision_traj_rep

    def encode_route_vision(self, route_data, route_assign_mat=None):
        if not self.use_vision or not self.use_route_vision:
            return None
        if route_data is None:
            return None
        feature_idx = list(self.route_vision_feature_idx)
        x_enc = route_data[:, :, feature_idx].float()
        if self.route_vision_use_log1p and 2 in feature_idx:
            interval_pos = feature_idx.index(2)
            x_enc[..., interval_pos] = torch.log1p(torch.clamp(x_enc[..., interval_pos], min=0.0))
        if self.route_vision_stats is not None:
            mean = torch.tensor(self.route_vision_stats['mean'], device=x_enc.device, dtype=x_enc.dtype)
            std = torch.tensor(self.route_vision_stats['std'], device=x_enc.device, dtype=x_enc.dtype)
            x_enc = (x_enc - mean) / std
        else:
            # Normalize known time-related features (weekday/minute/interval) when stats are unavailable.
            if 0 in feature_idx:
                x_enc[..., feature_idx.index(0)] = x_enc[..., feature_idx.index(0)] / 7.0
            if 1 in feature_idx:
                x_enc[..., feature_idx.index(1)] = x_enc[..., feature_idx.index(1)] / 1440.0
            if 2 in feature_idx:
                x_enc[..., feature_idx.index(2)] = x_enc[..., feature_idx.index(2)] / 100.0
        if route_assign_mat is not None:
            padding_mask = (route_assign_mat == self.vocab_size).unsqueeze(-1)
            x_enc = x_enc.masked_fill(padding_mask, 0.0)
        route_vision_traj_rep = self.route_vision_encoder(x_enc)
        route_vision_traj_rep = self.route_vision_proj_head(route_vision_traj_rep)
        return route_vision_traj_rep

    def compute_vision_rep(self, gps_data, route_data, gps_assign_mat=None, route_assign_mat=None):
        if not self.use_vision:
            return None
        vision_traj_rep = self.encode_vision(gps_data, gps_assign_mat=gps_assign_mat)
        if vision_traj_rep is None:
            return None
        route_vision_traj_rep = self.encode_route_vision(route_data, route_assign_mat=route_assign_mat)
        if route_vision_traj_rep is not None:
            vision_traj_rep = self.vision_fuse_route(torch.cat([vision_traj_rep, route_vision_traj_rep], dim=1))
        return vision_traj_rep

    def fuse_pair(self, base_rep, vision_rep, fuse_layer, gate_layer=None):
        if vision_rep is None:
            return base_rep
        fused_input = torch.cat([base_rep, vision_rep], dim=1)
        if gate_layer is None:
            return fuse_layer(fused_input)
        gate = torch.sigmoid(gate_layer(fused_input))
        gated_vision = vision_rep * gate
        return fuse_layer(torch.cat([base_rep, gated_vision], dim=1))

    def forward(self, route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat,
                gps_length):
        gps_road_rep, gps_traj_rep = self.encode_gps(gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length)
        route_road_rep, route_traj_rep = self.encode_route(route_data, route_assign_mat, masked_route_assign_mat)
        vision_traj_rep = self.encode_vision(gps_data, gps_assign_mat=masked_gps_assign_mat)
        route_vision_traj_rep = self.encode_route_vision(route_data, route_assign_mat=masked_route_assign_mat)
        if vision_traj_rep is not None and route_vision_traj_rep is not None:
            vision_traj_rep = self.vision_fuse_route(torch.cat([vision_traj_rep, route_vision_traj_rep], dim=1))
        if self.use_vision_pair_fuse:
            gps_traj_rep = self.fuse_pair(
                gps_traj_rep,
                vision_traj_rep,
                self.gps_vision_fuse,
                self.gps_vision_gate if self.use_vision_pair_gate else None,
            )
            if self.use_route_vision:
                route_traj_rep = self.fuse_pair(
                    route_traj_rep,
                    route_vision_traj_rep,
                    self.route_vision_fuse,
                    self.route_vision_gate if self.use_vision_pair_gate else None,
                )
            vision_traj_rep = None
        if vision_traj_rep is not None and self.use_vision_gate:
            gps_gate_rep = self.gps_gate_proj(gps_traj_rep)
            gate_in = torch.cat([gps_gate_rep, route_traj_rep, vision_traj_rep], dim=1)
            gate = torch.sigmoid(self.vision_gate(gate_in))
            fused_route_traj = torch.cat([route_traj_rep, vision_traj_rep * gate], dim=1)
            route_traj_rep = self.vision_fuse(fused_route_traj)
            vision_traj_rep = None
        elif vision_traj_rep is not None and self.vision_fuse_after_gru and not self.use_vision_in_joint:
            fused_route_traj = torch.cat([route_traj_rep, vision_traj_rep], dim=1)
            fused_gps_traj = torch.cat([gps_traj_rep, vision_traj_rep], dim=1)
            route_traj_rep = self.vision_fuse_after_gru_route(fused_route_traj)
            gps_traj_rep = self.vision_fuse_after_gru_gps(fused_gps_traj)
            vision_traj_rep = None
        gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep = self.encode_joint(
            route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat, vision_traj_rep=vision_traj_rep)
        if vision_traj_rep is not None and self.vision_fuse_after_joint and not self.use_vision_in_joint:
            fused_route_joint = torch.cat([route_traj_joint_rep, vision_traj_rep], dim=1)
            fused_gps_joint = torch.cat([gps_traj_joint_rep, vision_traj_rep], dim=1)
            route_traj_joint_rep = self.vision_fuse_after_joint_route(fused_route_joint)
            gps_traj_joint_rep = self.vision_fuse_after_joint_gps(fused_gps_joint)

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
        use_temporal_bias = getattr(self, 'add_temporal_bias', False) and temporal_mat is not None
        if use_temporal_bias:
            output = src
            temporal_mat = temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            for layer in self.transformer_encoder.layers:
                output, _ = layer(output, src_mask, src_key_padding_mask, temporal_mat=temporal_mat)
            return output

        if hasattr(self, 'transformer_encoder2'):
            return self.transformer_encoder2(src, src_mask, src_key_padding_mask)

        return self.transformer_encoder(src, src_mask, src_key_padding_mask)


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


class BiCrossModalFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1, num_layers=1):
        super(BiCrossModalFusion, self).__init__()
        self.layers = nn.ModuleList([
            BiCrossModalLayer(hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, gps_tokens, route_tokens, gps_key_padding_mask=None, route_key_padding_mask=None):
        out_gps, out_route = gps_tokens, route_tokens
        for layer in self.layers:
            out_gps, out_route = layer(
                out_gps,
                out_route,
                gps_key_padding_mask=gps_key_padding_mask,
                route_key_padding_mask=route_key_padding_mask,
            )
        return out_gps, out_route


class BiCrossModalLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(BiCrossModalLayer, self).__init__()
        self.gps_to_route = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.route_to_gps = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.gps_norm1 = nn.LayerNorm(hidden_size)
        self.route_norm1 = nn.LayerNorm(hidden_size)
        self.gps_norm2 = nn.LayerNorm(hidden_size)
        self.route_norm2 = nn.LayerNorm(hidden_size)

        self.gps_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.route_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, gps_tokens, route_tokens, gps_key_padding_mask=None, route_key_padding_mask=None):
        gps_ctx, _ = self.gps_to_route(
            query=gps_tokens,
            key=route_tokens,
            value=route_tokens,
            key_padding_mask=route_key_padding_mask,
            need_weights=False,
        )
        route_ctx, _ = self.route_to_gps(
            query=route_tokens,
            key=gps_tokens,
            value=gps_tokens,
            key_padding_mask=gps_key_padding_mask,
            need_weights=False,
        )

        gps_tokens = self.gps_norm1(gps_tokens + self.dropout(gps_ctx))
        route_tokens = self.route_norm1(route_tokens + self.dropout(route_ctx))

        gps_tokens = self.gps_norm2(gps_tokens + self.dropout(self.gps_ffn(gps_tokens)))
        route_tokens = self.route_norm2(route_tokens + self.dropout(self.route_ffn(route_tokens)))
        return gps_tokens, route_tokens
