import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .fds import *
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 5-Layer ANN Regressor


class ANN(nn.Module):
    def __init__(self, input_size, FDS=None):
        super(ANN, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 5000),
            nn.ReLU(),
            nn.BatchNorm1d(5000),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5000, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.FDS = FDS

    def forward(self, x, y=None, epoch=None, device=None):
        x = self.encoder(x)
        if self.FDS is not None:
            if epoch >= 1:
                self.FDS.cpu()
                x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
                y = y.to(device)
        x = self.decoder(x.to(device))
        return x

# MMHRP


class GATU(nn.Module):
    def __init__(self, node_feature_num, channels, heads):
        super(GATU, self).__init__()
        self.conv1 = pyg_nn.GATConv(node_feature_num, channels[0], heads=heads)
        self.norm1 = nn.BatchNorm1d(channels[0] * heads)
        self.conv2 = pyg_nn.GATConv(
            channels[0] * heads, channels[1], heads=heads)

    def forward(self, data):
        x = data["x"].clone().detach().float()
        edge_index = data["edge_index"].clone().detach()
        batch = data["batch"].clone().detach()

        # GAT
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x = torch.cat([x1, x2], 1)

        # Pooling
        x_mean = pyg_nn.global_mean_pool(x, batch=batch)
        x_max = pyg_nn.global_max_pool(x, batch=batch)
        x = torch.cat([x_mean, x_max], 1)

        return x

# GAT


class GAT(nn.Module):
    def __init__(self, node_feature_num, EncoderChannels, heads, DecoderChannels):
        super(GAT, self).__init__()
        assert len(EncoderChannels) == 2
        assert len(DecoderChannels) == 2
        self.encoder = GATU(node_feature_num, EncoderChannels, heads)
        self.emb_size = 0
        for i in EncoderChannels:
            self.emb_size += i * heads
        self.emb_size = self.emb_size * 2
        self.decoder = nn.Sequential(
            nn.Linear(self.emb_size, DecoderChannels[0]),
            nn.ReLU(),
            nn.Linear(DecoderChannels[0], DecoderChannels[1]),
            nn.ReLU(),
            nn.Linear(DecoderChannels[1], 1)
        )

    def forward(self, data):
        # Encoder
        x = self.encoder(data)

        # Decoder
        x = self.decoder(x)

        return x


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input layout: batch, sequence, embedding
            bidirectional=True
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.device = device

    def init_hidden(self, x):
        return torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

    def forward(self, x):
        h0 = self.init_hidden(x)
        # x is input, size (batch, seq_len, input_size)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    
class MMHRP_GCL(nn.Module):
    def __init__(self,
                 GraphEncoder: dict,
                 TextEncoder: dict,
                 ModalityAlignment: dict,
                 Decoder: dict,
                 emb_size: int = 128, # embedding vector size for each encoder
                 device: str = torch.device('cuda'),
                 FDS=None
                 ):
        super(MMHRP_GCL, self).__init__()

        self.device = device
        self.GraphEncoder = GraphEncoder
        self.TextEncoder = TextEncoder
        self.FDS = FDS

        # 1.Encoder Parser
        # 1.1 GraphEncoder Parser
        Graph_params = ["NodeFeatNum", "Channels", "Heads"]
        for key in GraphEncoder.keys():
            if key not in Graph_params:
                raise Exception("%s is not the param in Graph Encoder")

            if key == "NodeFeatNum":
                NodeFeatNum = GraphEncoder[key]  # 8
            if key == "Channels":
                GAT_Channels = GraphEncoder[key]  # [32, 64]
                assert len(GAT_Channels) == 2
            if key == "Heads":
                GAT_Heads = GraphEncoder[key]  # 4

        # GATU output size
        GATU_OutSize = 0
        for i in GAT_Channels:
            GATU_OutSize += i * GAT_Heads
        self.GATU_OutSize = GATU_OutSize * 2

        # GraphEncoder
        self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        # GATU for Reactants and Products
        self.ReaProEncoder = nn.Sequential(
            self.GATU_ReaPro,
            nn.Linear(self.GATU_OutSize, emb_size),
        )
        # GATU for Catalysts and Solvents
        self.CatSolEncoder = nn.Sequential(
            self.GATU_CatSol,
            nn.Linear(self.GATU_OutSize, emb_size),
        )

        # 1.2 TextEncoder Parser
        Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
        for key in TextEncoder.keys():
            if key not in Text_params:
                raise Exception("%s is not the param in Text Encoder")

            if key == "SmiFeatNum":
                self.SmiFeatNum = TextEncoder[key]  # 128
            if key == "Heads":
                Trans_Heads = TextEncoder[key]  # 4
            if key == "BigruChannels":
                BigruChannels = TextEncoder[key]  # [128, 128]
                assert len(BigruChannels) == 2
                bigru_input, bigru_hidden = BigruChannels
            if key == "BigruNumlayer":
                bigru_num_layers = TextEncoder[key]  # 2

        # TextEncoder
        self.trans = nn.TransformerEncoderLayer(d_model=self.SmiFeatNum, nhead=Trans_Heads, batch_first=True,
                                                norm_first=True)
        self.bigru = BiGRU(bigru_input, bigru_hidden, bigru_num_layers, emb_size, device)
        self.RxnSmiEncoder = nn.Sequential(
            self.trans,
            self.bigru,
        )

        # 3. Modality Alignment Parser
        MA_params = ["Heads"]
        for key in ModalityAlignment.keys():
            if key not in MA_params:
                raise Exception("%s is not the param in ModalityAlignment")

            if key == "Heads":
                MA_Heads = ModalityAlignment[key]  # 4

        self.MA = nn.TransformerEncoderLayer(d_model=emb_size * 3, nhead=MA_Heads)

        # 4. Decoder Parser
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception("%s is not the param in Decoder")

            if key == "Channels":
                Decoder_Channels = Decoder[key]  # [1000, 500, 100]
                assert len(Decoder_Channels) == 3

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(emb_size * 3, Decoder_Channels[0]),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),
            

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    def forward(self, x, y=None, epoch=None, device=None):
        # import data
        ReaPro_data, CatSol_data, RxnSmi = x
        # Graph Modality
        graph_emb = torch.cat([self.ReaProEncoder(ReaPro_data),
                                   self.CatSolEncoder(CatSol_data)], dim=1)
        # Text Modality
        text_embed = self.RxnSmiEncoder(RxnSmi)
        x = torch.cat([graph_emb, text_embed], dim=1)


        # Modality Alignment
        x = self.MA(x)

        # FDS operation
        if self.FDS is not None:
            if epoch >= 1:
                self.FDS.cpu()
                x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
                y = y.to(device)

        # Decoder
        x = self.decoder(x)

        return x

class MMHRP_GCL_tri(nn.Module):
    def __init__(self,
                 GraphEncoder: dict,
                 TextEncoder: dict,
                 ConditionEncoder: dict,
                 ModalityAlignment: dict,
                 Decoder: dict,
                 emb_size: int = 128, # embedding vector size for each encoder
                 device: str = torch.device('cuda'),
                 FDS=None
                 ):
        super(MMHRP_GCL_tri, self).__init__()

        self.device = device
        self.GraphEncoder = GraphEncoder
        self.TextEncoder = TextEncoder
        self.FDS = FDS

        # 1.Encoder Parser
        # 1.1 GraphEncoder Parser
        Graph_params = ["NodeFeatNum", "Channels", "Heads"]
        for key in GraphEncoder.keys():
            if key not in Graph_params:
                raise Exception("%s is not the param in Graph Encoder")

            if key == "NodeFeatNum":
                NodeFeatNum = GraphEncoder[key]  # 8
            if key == "Channels":
                GAT_Channels = GraphEncoder[key]  # [32, 64]
                assert len(GAT_Channels) == 2
            if key == "Heads":
                GAT_Heads = GraphEncoder[key]  # 4

        # GATU output size
        GATU_OutSize = 0
        for i in GAT_Channels:
            GATU_OutSize += i * GAT_Heads
        self.GATU_OutSize = GATU_OutSize * 2

        # GraphEncoder
        self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        # GATU for Reactants and Products
        self.ReaProEncoder = nn.Sequential(
            self.GATU_ReaPro,
            nn.Linear(self.GATU_OutSize, emb_size),
        )
        # GATU for Catalysts and Solvents
        self.CatSolEncoder = nn.Sequential(
            self.GATU_CatSol,
            nn.Linear(self.GATU_OutSize, emb_size),
        )

        # 1.2 TextEncoder Parser
        Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
        for key in TextEncoder.keys():
            if key not in Text_params:
                raise Exception("%s is not the param in Text Encoder")

            if key == "SmiFeatNum":
                self.SmiFeatNum = TextEncoder[key]  # 128
            if key == "Heads":
                Trans_Heads = TextEncoder[key]  # 4
            if key == "BigruChannels":
                BigruChannels = TextEncoder[key]  # [128, 128]
                assert len(BigruChannels) == 2
                bigru_input, bigru_hidden = BigruChannels
            if key == "BigruNumlayer":
                bigru_num_layers = TextEncoder[key]  # 2

        # TextEncoder
        self.trans = nn.TransformerEncoderLayer(d_model=self.SmiFeatNum, nhead=Trans_Heads, batch_first=True,
                                                norm_first=True)
        self.bigru = BiGRU(bigru_input, bigru_hidden, bigru_num_layers, emb_size, device)
        self.RxnSmiEncoder = nn.Sequential(
            self.trans,
            self.bigru,
        )

        # 1.3 TextEncoder Parser
        Cond_params = ["InputDim", "HiddenDim", "OutDim"]
        for key in ConditionEncoder.keys():
            if key not in Cond_params:
                raise Exception("%s is not the param in Condition Encoder" % key)
            if key == "InputDim":
                Cond_InputDim = ConditionEncoder[key]
            if key == "HiddenDim":
                Cond_HiddenDim = ConditionEncoder[key]
            if key == "OutDim":
                Cond_OutDim = ConditionEncoder[key]

        self.CondEncoder = nn.Sequential(
            nn.Linear(Cond_InputDim, Cond_HiddenDim),
            nn.ReLU(),
            nn.Linear(Cond_HiddenDim, Cond_OutDim),
            nn.ReLU()
        )

        
        # 3. Modality Alignment Parser
        MA_params = ["Heads"]
        for key in ModalityAlignment.keys():
            if key not in MA_params:
                raise Exception("%s is not the param in ModalityAlignment")

            if key == "Heads":
                MA_Heads = ModalityAlignment[key] 

        # Total embedding dimension: graph(2*emb_size) + text(emb_size) + condition(OutDim)
        total_emb = emb_size*2 + emb_size + Cond_OutDim

        self.MA = nn.TransformerEncoderLayer(d_model=total_emb, nhead=MA_Heads)

        # 4. Decoder Parser
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception("%s is not the param in Decoder")

            if key == "Channels":
                Decoder_Channels = Decoder[key]  # [1000, 500, 100]
                assert len(Decoder_Channels) == 3

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_emb, Decoder_Channels[0]),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),
            

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    def forward(self, x, y=None, epoch=None, device=None):
        # import data
        ReaPro_data, CatSol_data, RxnSmi, Cond_data = x
        # Graph Modality
        graph_emb = torch.cat([self.ReaProEncoder(ReaPro_data),
                                   self.CatSolEncoder(CatSol_data)], dim=1)
        
        # Text Modality
        text_embed = self.RxnSmiEncoder(RxnSmi)
        # x = torch.cat([graph_emb, text_embed], dim=1)

        # Condition Modality
        cond_emb = self.CondEncoder(Cond_data)
        
        # Three-modality fusion
        x = torch.cat([graph_emb, text_embed, cond_emb], dim=1)
        
        # Modality Alignment
        x = self.MA(x)

        # FDS operation
        if self.FDS is not None:
            if epoch >= 1:
                self.FDS.cpu()
                x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
                y = y.to(device)

        # Decoder
        x = self.decoder(x)

        return x

    
    
# class MMHRP_GCL_tri_MoPE_pdall(nn.Module):
#     def __init__(self,
#                  GraphEncoder: dict,
#                  TextEncoder: dict,
#                  ConditionEncoder: dict,
#                  Decoder: dict,
#                  emb_size: int = 128,
#                  device: str = torch.device('cuda'),
#                  FDS=None
#                  ):
#         super(MMHRP_GCL_tri_MoPE_pdall, self).__init__()

#         self.device = device
#         self.FDS = FDS
#         self.emb_size = emb_size

#         # =====================================================
#         # 1. Graph encoder (primary modality)
#         # =====================================================
#         Graph_params = ["NodeFeatNum", "Channels", "Heads"]
#         for key in GraphEncoder.keys():
#             if key not in Graph_params:
#                 raise Exception(f"{key} is not a param in Graph Encoder")
#             if key == "NodeFeatNum":
#                 NodeFeatNum = GraphEncoder[key]
#             if key == "Channels":
#                 GAT_Channels = GraphEncoder[key]
#                 assert len(GAT_Channels) == 2
#             if key == "Heads":
#                 GAT_Heads = GraphEncoder[key]

#         GATU_OutSize = sum([c * GAT_Heads for c in GAT_Channels])
#         self.GATU_OutSize = GATU_OutSize * 2

#         self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
#         self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)

#         self.ReaProEncoder = nn.Sequential(
#             self.GATU_ReaPro,
#             nn.Linear(self.GATU_OutSize, emb_size),
#         )
#         self.CatSolEncoder = nn.Sequential(
#             self.GATU_CatSol,
#             nn.Linear(self.GATU_OutSize, emb_size),
#         )

#         # =====================================================
#         # 2. Text Encoder（SMILES）
#         # =====================================================
#         Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
#         for key in TextEncoder.keys():
#             if key not in Text_params:
#                 raise Exception(f"{key} is not a param in Text Encoder")
#             if key == "SmiFeatNum":
#                 self.SmiFeatNum = TextEncoder[key]
#             if key == "Heads":
#                 Trans_Heads = TextEncoder[key]
#             if key == "BigruChannels":
#                 BigruChannels = TextEncoder[key]
#                 bigru_input, bigru_hidden = BigruChannels
#             if key == "BigruNumlayer":
#                 bigru_num_layers = TextEncoder[key]

#         self.trans = nn.TransformerEncoderLayer(
#             d_model=self.SmiFeatNum,
#             nhead=Trans_Heads,
#             batch_first=True,
#             norm_first=True
#         )
#         self.bigru = BiGRU(
#             bigru_input,
#             bigru_hidden,
#             bigru_num_layers,
#             emb_size,
#             device
#         )

#         self.RxnSmiEncoder = nn.Sequential(
#             self.trans,
#             self.bigru,
#         )

#         # =====================================================
#         # 3. Numerical condition encoder
#         # =====================================================
#         Cond_params = ["InputDim", "HiddenDim", "OutDim"]
#         for key in ConditionEncoder.keys():
#             if key not in Cond_params:
#                 raise Exception(f"{key} is not a param in Condition Encoder")
#             if key == "InputDim":
#                 Cond_InputDim = ConditionEncoder[key]
#             if key == "HiddenDim":
#                 Cond_HiddenDim = ConditionEncoder[key]
#             if key == "OutDim":
#                 Cond_OutDim = ConditionEncoder[key]

#         self.CondEncoder = nn.Sequential(
#             nn.Linear(Cond_InputDim, Cond_HiddenDim),
#             nn.ReLU(),
#             nn.Linear(Cond_HiddenDim, Cond_OutDim),
#             # nn.ReLU()
#         )

#         # =====================================================
#         # 4. MoPE（Prompt-level Experts + Conditional Routing）
#         # =====================================================
#         self.num_experts = 2
#         self.prompt_len = 3
#         self.prompt_dim = emb_size

#         # ---- Prompt Experts ----
#         self.prompt_experts = nn.Parameter(
#             torch.randn(self.num_experts, self.prompt_len, self.prompt_dim)
#         )

#         # ---- Router（Graph + Text + Condition）----
#         self.router = nn.Sequential(
#             nn.Linear(emb_size * 3 + Cond_OutDim, emb_size),
#             nn.ReLU(),
#             nn.Linear(emb_size, self.num_experts),
#             nn.Softmax(dim=1)
#         )

#         # ---- Condition → Prompt Mapping ----
#         self.cond_to_prompt = nn.Linear(
#             Cond_OutDim,
#             self.prompt_len * self.prompt_dim
#         )

#         # =====================================================
#         # 5. Decoder
#         # =====================================================
#         Decoder_params = ["Channels"]
#         for key in Decoder.keys():
#             if key not in Decoder_params:
#                 raise Exception(f"{key} is not a param in Decoder")
#             if key == "Channels":
#                 Decoder_Channels = Decoder[key]
#                 assert len(Decoder_Channels) == 3

#         self.decoder = nn.Sequential(
#             nn.Linear(emb_size, Decoder_Channels[0]),
#             nn.ReLU(),
#             nn.BatchNorm1d(Decoder_Channels[0]),

#             nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
#             nn.ReLU(),

#             nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
#             nn.ReLU(),

#             nn.Linear(Decoder_Channels[2], 1)
#         )

#     # =====================================================
#     # Forward
#     # =====================================================
#     def forward(self, x, y=None, epoch=None, device=None):
#         if device is None:
#             device = self.device

#         ReaPro_data, CatSol_data, RxnSmi, Cond_data = x

#         # -------- Graph --------
#         graph_emb = torch.cat([
#             self.ReaProEncoder(ReaPro_data),
#             self.CatSolEncoder(CatSol_data)
#         ], dim=1)  # [B, 2*emb]

#         # -------- Text --------
#         text_emb = self.RxnSmiEncoder(RxnSmi)  # [B, emb]

#         # -------- Condition --------
#         cond_emb = self.CondEncoder(Cond_data)  # [B, Cond_OutDim]

#         # =====================================================
#         # MoPE Routing（Pd：Graph + Text + Condition）
#         # =====================================================
#         router_input = torch.cat(
#             [graph_emb, text_emb, cond_emb],
#             dim=1
#         )

#         expert_weights = self.router(router_input)  # [B, k]

#         base_prompt = torch.einsum(
#             "bk,kld->bld",
#             expert_weights,
#             self.prompt_experts
#         )

#         cond_prompt = self.cond_to_prompt(cond_emb).view(
#             cond_emb.size(0),
#             self.prompt_len,
#             self.prompt_dim
#         )

#         dynamic_prompt = base_prompt + cond_prompt  # [B, L, D]

#         # =====================================================
#         # Prompt injection as virtual nodes
#         # =====================================================
#         graph_tokens = graph_emb.view(
#             graph_emb.size(0), 2, self.prompt_dim
#         )

#         graph_with_prompt = torch.cat(
#             [dynamic_prompt, graph_tokens],
#             dim=1
#         )

#         x = graph_with_prompt.mean(dim=1)

#         # -------- FDS --------
#         if self.FDS is not None and y is not None and epoch is not None and epoch >= 1:
#             self.FDS.cpu()
#             x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
#             y = y.to(device)

#         # -------- Decoder --------
#         out = self.decoder(x)
#         return out
 
    
    
    
class MMHRP_GCL_tri_MoPE_pdall(nn.Module):
    def __init__(self,
                 GraphEncoder: dict,
                 TextEncoder: dict,
                 ConditionEncoder: dict,
                 Decoder: dict,
                 emb_size: int = 128,
                 device: str = torch.device('cuda'),
                 FDS=None
                 ):
        super(MMHRP_GCL_tri_MoPE_pdall, self).__init__()

        self.device = device
        self.FDS = FDS
        self.emb_size = emb_size

        # =====================================================
        # 1. Graph encoder (primary modality)
        # =====================================================
        Graph_params = ["NodeFeatNum", "Channels", "Heads"]
        for key in GraphEncoder.keys():
            if key not in Graph_params:
                raise Exception(f"{key} is not a param in Graph Encoder")
            if key == "NodeFeatNum":
                NodeFeatNum = GraphEncoder[key]
            if key == "Channels":
                GAT_Channels = GraphEncoder[key]
                assert len(GAT_Channels) == 2
            if key == "Heads":
                GAT_Heads = GraphEncoder[key]

        GATU_OutSize = sum([c * GAT_Heads for c in GAT_Channels])
        self.GATU_OutSize = GATU_OutSize * 2

        self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)

        self.ReaProEncoder = nn.Sequential(
            self.GATU_ReaPro,
            nn.Linear(self.GATU_OutSize, emb_size),
        )
        self.CatSolEncoder = nn.Sequential(
            self.GATU_CatSol,
            nn.Linear(self.GATU_OutSize, emb_size),
        )

        # =====================================================
        # 2. Text Encoder（SMILES）
        # =====================================================
        Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
        for key in TextEncoder.keys():
            if key not in Text_params:
                raise Exception(f"{key} is not a param in Text Encoder")
            if key == "SmiFeatNum":
                self.SmiFeatNum = TextEncoder[key]
            if key == "Heads":
                Trans_Heads = TextEncoder[key]
            if key == "BigruChannels":
                BigruChannels = TextEncoder[key]
                bigru_input, bigru_hidden = BigruChannels
            if key == "BigruNumlayer":
                bigru_num_layers = TextEncoder[key]

        self.trans = nn.TransformerEncoderLayer(
            d_model=self.SmiFeatNum,
            nhead=Trans_Heads,
            batch_first=True,
            norm_first=True
        )
        self.bigru = BiGRU(
            bigru_input,
            bigru_hidden,
            bigru_num_layers,
            emb_size,
            device
        )

        self.RxnSmiEncoder = nn.Sequential(
            self.trans,
            self.bigru,
        )

        # =====================================================
        # 3. Numerical condition encoder
        # =====================================================
        Cond_params = ["InputDim", "HiddenDim", "OutDim"]
        for key in ConditionEncoder.keys():
            if key not in Cond_params:
                raise Exception(f"{key} is not a param in Condition Encoder")
            if key == "InputDim":
                Cond_InputDim = ConditionEncoder[key]
            if key == "HiddenDim":
                Cond_HiddenDim = ConditionEncoder[key]
            if key == "OutDim":
                Cond_OutDim = ConditionEncoder[key]

        self.CondEncoder = nn.Sequential(
            nn.Linear(Cond_InputDim, Cond_HiddenDim),
            nn.ReLU(),
            nn.Linear(Cond_HiddenDim, Cond_OutDim),
            # nn.ReLU()
        )

        # =====================================================
        # 4. MoPE（Prompt-level Experts + Conditional Routing）
        # =====================================================
        self.num_experts = 2
        self.prompt_len = 3
        self.prompt_dim = emb_size

        # ---- Prompt Experts ----
        self.prompt_experts = nn.Parameter(
            torch.randn(self.num_experts, self.prompt_len, self.prompt_dim)
        )

        # ---- Router（Graph + Text + Condition）----
        self.router = nn.Sequential(
            nn.Linear(emb_size * 3 + Cond_OutDim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, self.num_experts),
            nn.Softmax(dim=1)
        )

        # ---- Condition → Prompt Mapping ----
        self.cond_to_prompt = nn.Linear(
            Cond_OutDim,
            self.prompt_len * self.prompt_dim
        )

        # =====================================================
        # 5. Decoder
        # =====================================================
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception(f"{key} is not a param in Decoder")
            if key == "Channels":
                Decoder_Channels = Decoder[key]
                assert len(Decoder_Channels) == 3

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, Decoder_Channels[0]),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x, y=None, epoch=None, device=None):
        if device is None:
            device = self.device

        ReaPro_data, CatSol_data, RxnSmi, Cond_data = x

        # -------- Graph --------
        graph_emb = torch.cat([
            self.ReaProEncoder(ReaPro_data),
            self.CatSolEncoder(CatSol_data)
        ], dim=1)  # [B, 2*emb]

        # -------- Text --------
        text_emb = self.RxnSmiEncoder(RxnSmi)  # [B, emb]

        # -------- Condition --------
        cond_emb = self.CondEncoder(Cond_data)  # [B, Cond_OutDim]

        # =====================================================
        # MoPE Routing（Pd：Graph + Text + Condition）
        # =====================================================
        router_input = torch.cat(
            [graph_emb, text_emb, cond_emb],
            dim=1
        )

        expert_weights = self.router(router_input)  # [B, k]

        base_prompt = torch.einsum(
            "bk,kld->bld",
            expert_weights,
            self.prompt_experts
        )

        cond_prompt = self.cond_to_prompt(cond_emb).view(
            cond_emb.size(0),
            self.prompt_len,
            self.prompt_dim
        )

        dynamic_prompt = base_prompt + cond_prompt  # [B, L, D]

        # =====================================================
        # Prompt injection as virtual nodes
        # =====================================================
        graph_tokens = graph_emb.view(
            graph_emb.size(0), 2, self.prompt_dim
        )

        graph_with_prompt = torch.cat(
            [dynamic_prompt, graph_tokens],
            dim=1
        )

        x = graph_with_prompt.mean(dim=1)

        # -------- FDS --------
        if self.FDS is not None and y is not None and epoch is not None and epoch >= 1:
            self.FDS.cpu()
            x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
            y = y.to(device)

        # -------- Decoder --------
        out = self.decoder(x)
        return out
        
# Explanation analysis
# class MMHRP_GCL_tri_MoPE_pdall(nn.Module):
#     def __init__(self,
#                  GraphEncoder: dict,
#                  TextEncoder: dict,
#                  ConditionEncoder: dict,
#                  Decoder: dict,
#                  emb_size: int = 128,
#                  device: str = torch.device('cuda'),
#                  FDS=None
#                  ):
#         super(MMHRP_GCL_tri_MoPE_pdall, self).__init__()

#         self.device = device
#         self.FDS = FDS
#         self.emb_size = emb_size

#         # =====================================================
#         # 1. Graph encoder (primary modality)
#         # =====================================================
#         Graph_params = ["NodeFeatNum", "Channels", "Heads"]
#         for key in GraphEncoder.keys():
#             if key not in Graph_params:
#                 raise Exception(f"{key} is not a param in Graph Encoder")
#             if key == "NodeFeatNum":
#                 NodeFeatNum = GraphEncoder[key]
#             if key == "Channels":
#                 GAT_Channels = GraphEncoder[key]
#                 assert len(GAT_Channels) == 2
#             if key == "Heads":
#                 GAT_Heads = GraphEncoder[key]

#         GATU_OutSize = sum([c * GAT_Heads for c in GAT_Channels])
#         self.GATU_OutSize = GATU_OutSize * 2

#         self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
#         self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)

#         self.ReaProEncoder = nn.Sequential(
#             self.GATU_ReaPro,
#             nn.Linear(self.GATU_OutSize, emb_size),
#         )
#         self.CatSolEncoder = nn.Sequential(
#             self.GATU_CatSol,
#             nn.Linear(self.GATU_OutSize, emb_size),
#         )

#         # =====================================================
#         # 2. Text Encoder（SMILES）
#         # =====================================================
#         Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
#         for key in TextEncoder.keys():
#             if key not in Text_params:
#                 raise Exception(f"{key} is not a param in Text Encoder")
#             if key == "SmiFeatNum":
#                 self.SmiFeatNum = TextEncoder[key]
#             if key == "Heads":
#                 Trans_Heads = TextEncoder[key]
#             if key == "BigruChannels":
#                 BigruChannels = TextEncoder[key]
#                 bigru_input, bigru_hidden = BigruChannels
#             if key == "BigruNumlayer":
#                 bigru_num_layers = TextEncoder[key]

#         self.trans = nn.TransformerEncoderLayer(
#             d_model=self.SmiFeatNum,
#             nhead=Trans_Heads,
#             batch_first=True,
#             norm_first=True
#         )
#         self.bigru = BiGRU(
#             bigru_input,
#             bigru_hidden,
#             bigru_num_layers,
#             emb_size,
#             device
#         )

#         self.RxnSmiEncoder = nn.Sequential(
#             self.trans,
#             self.bigru,
#         )

#         # =====================================================
#         # 3. Numerical condition encoder
#         # =====================================================
#         Cond_params = ["InputDim", "HiddenDim", "OutDim"]
#         for key in ConditionEncoder.keys():
#             if key not in Cond_params:
#                 raise Exception(f"{key} is not a param in Condition Encoder")
#             if key == "InputDim":
#                 Cond_InputDim = ConditionEncoder[key]
#             if key == "HiddenDim":
#                 Cond_HiddenDim = ConditionEncoder[key]
#             if key == "OutDim":
#                 Cond_OutDim = ConditionEncoder[key]

#         self.CondEncoder = nn.Sequential(
#             nn.Linear(Cond_InputDim, Cond_HiddenDim),
#             nn.ReLU(),
#             nn.Linear(Cond_HiddenDim, Cond_OutDim),
#             # nn.ReLU()
#         )

#         # =====================================================
#         # 4. MoPE（Prompt-level Experts + Conditional Routing）
#         # =====================================================
#         self.num_experts = 2
#         self.prompt_len = 3
#         self.prompt_dim = emb_size

#         # ---- Prompt Experts ----
#         self.prompt_experts = nn.Parameter(
#             torch.randn(self.num_experts, self.prompt_len, self.prompt_dim)
#         )

#         # ---- Router（Graph + Text + Condition）----
#         self.router = nn.Sequential(
#             nn.Linear(emb_size * 3 + Cond_OutDim, emb_size),
#             nn.ReLU(),
#             nn.Linear(emb_size, self.num_experts),
#             nn.Softmax(dim=1)
#         )

#         # ---- Condition → Prompt Mapping ----
#         self.cond_to_prompt = nn.Linear(
#             Cond_OutDim,
#             self.prompt_len * self.prompt_dim
#         )

#         # =====================================================
#         # 5. Decoder
#         # =====================================================
#         Decoder_params = ["Channels"]
#         for key in Decoder.keys():
#             if key not in Decoder_params:
#                 raise Exception(f"{key} is not a param in Decoder")
#             if key == "Channels":
#                 Decoder_Channels = Decoder[key]
#                 assert len(Decoder_Channels) == 3

#         self.decoder = nn.Sequential(
#             nn.Linear(emb_size, Decoder_Channels[0]),
#             nn.ReLU(),
#             nn.BatchNorm1d(Decoder_Channels[0]),

#             nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
#             nn.ReLU(),

#             nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
#             nn.ReLU(),

#             nn.Linear(Decoder_Channels[2], 1)
#         )

#     # =====================================================
#     # Forward
#     # =====================================================
#     def forward(self, x, y=None, epoch=None, device=None):
#         if device is None:
#             device = self.device

#         ReaPro_data, CatSol_data, RxnSmi, Cond_data = x

#         # -------- Graph --------
#         graph_emb = torch.cat([
#             self.ReaProEncoder(ReaPro_data),
#             self.CatSolEncoder(CatSol_data)
#         ], dim=1)  # [B, 2*emb]

#         # -------- Text --------
#         text_emb = self.RxnSmiEncoder(RxnSmi)  # [B, emb]

#         # -------- Condition --------
#         cond_emb = self.CondEncoder(Cond_data)  # [B, Cond_OutDim]

#         # =====================================================
#         # MoPE Routing（Pd：Graph + Text + Condition）
#         # =====================================================
#         router_input = torch.cat(
#             [graph_emb, text_emb, cond_emb],
#             dim=1
#         )

#         expert_weights = self.router(router_input)  # [B, k]

#         base_prompt = torch.einsum(
#             "bk,kld->bld",
#             expert_weights,
#             self.prompt_experts
#         )

#         cond_prompt = self.cond_to_prompt(cond_emb).view(
#             cond_emb.size(0),
#             self.prompt_len,
#             self.prompt_dim
#         )

#         dynamic_prompt = base_prompt + cond_prompt  # [B, L, D]

#         # =====================================================
#         # Prompt injection as virtual nodes
#         # =====================================================
#         graph_tokens = graph_emb.view(
#             graph_emb.size(0), 2, self.prompt_dim
#         )

#         graph_with_prompt = torch.cat(
#             [dynamic_prompt, graph_tokens],
#             dim=1
#         )

#         x = graph_with_prompt.mean(dim=1)

#         # -------- FDS --------
#         if self.FDS is not None and y is not None and epoch is not None and epoch >= 1:
#             self.FDS.cpu()
#             x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
#             y = y.to(device)

#         # -------- Decoder --------
#         out = self.decoder(x)
#         return out
    
    
# class MMHRP_GCL_tri_MoPE_pdall_explain(nn.Module):
#     def __init__(self,
#                  GraphEncoder: dict,
#                  TextEncoder: dict,
#                  ConditionEncoder: dict,
#                  Decoder: dict,
#                  emb_size: int = 128,
#                  device: str = torch.device('cuda'),
#                  FDS=None
#                  ):
#         super(MMHRP_GCL_tri_MoPE_pdall_explain, self).__init__()

#         self.device = device
#         self.FDS = FDS
#         self.emb_size = emb_size

#         # =====================================================
#         # 1. Graph encoder (primary modality)
#         # =====================================================
#         Graph_params = ["NodeFeatNum", "Channels", "Heads"]
#         for key in GraphEncoder.keys():
#             if key not in Graph_params:
#                 raise Exception(f"{key} is not a param in Graph Encoder")
#             if key == "NodeFeatNum":
#                 NodeFeatNum = GraphEncoder[key]
#             if key == "Channels":
#                 GAT_Channels = GraphEncoder[key]
#                 assert len(GAT_Channels) == 2
#             if key == "Heads":
#                 GAT_Heads = GraphEncoder[key]

#         GATU_OutSize = sum([c * GAT_Heads for c in GAT_Channels])
#         self.GATU_OutSize = GATU_OutSize * 2

#         self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
#         self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)

#         self.ReaProEncoder = nn.Sequential(
#             self.GATU_ReaPro,
#             nn.Linear(self.GATU_OutSize, emb_size),
#         )
#         self.CatSolEncoder = nn.Sequential(
#             self.GATU_CatSol,
#             nn.Linear(self.GATU_OutSize, emb_size),
#         )

#         # =====================================================
#         # 2. Text Encoder（SMILES）
#         # =====================================================
#         Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
#         for key in TextEncoder.keys():
#             if key not in Text_params:
#                 raise Exception(f"{key} is not a param in Text Encoder")
#             if key == "SmiFeatNum":
#                 self.SmiFeatNum = TextEncoder[key]
#             if key == "Heads":
#                 Trans_Heads = TextEncoder[key]
#             if key == "BigruChannels":
#                 BigruChannels = TextEncoder[key]
#                 bigru_input, bigru_hidden = BigruChannels
#             if key == "BigruNumlayer":
#                 bigru_num_layers = TextEncoder[key]

#         self.trans = nn.TransformerEncoderLayer(
#             d_model=self.SmiFeatNum,
#             nhead=Trans_Heads,
#             batch_first=True,
#             norm_first=True
#         )
#         self.bigru = BiGRU(
#             bigru_input,
#             bigru_hidden,
#             bigru_num_layers,
#             emb_size,
#             device
#         )

#         self.RxnSmiEncoder = nn.Sequential(
#             self.trans,
#             self.bigru,
#         )

#         # =====================================================
#         # 3. Numerical condition encoder
#         # =====================================================
#         Cond_params = ["InputDim", "HiddenDim", "OutDim"]
#         for key in ConditionEncoder.keys():
#             if key not in Cond_params:
#                 raise Exception(f"{key} is not a param in Condition Encoder")
#             if key == "InputDim":
#                 Cond_InputDim = ConditionEncoder[key]
#             if key == "HiddenDim":
#                 Cond_HiddenDim = ConditionEncoder[key]
#             if key == "OutDim":
#                 Cond_OutDim = ConditionEncoder[key]

#         self.CondEncoder = nn.Sequential(
#             nn.Linear(Cond_InputDim, Cond_HiddenDim),
#             nn.ReLU(),
#             nn.Linear(Cond_HiddenDim, Cond_OutDim),
#             # nn.ReLU()
#         )

#         # =====================================================
#         # 4. MoPE（Prompt-level Experts + Conditional Routing）
#         # =====================================================
#         self.num_experts = 2
#         self.prompt_len = 3
#         self.prompt_dim = emb_size

#         # ---- Prompt Experts ----
#         self.prompt_experts = nn.Parameter(
#             torch.randn(self.num_experts, self.prompt_len, self.prompt_dim)
#         )

#         # ---- Router（Graph + Text + Condition）----
#         self.router = nn.Sequential(
#             nn.Linear(emb_size * 3 + Cond_OutDim, emb_size),
#             nn.ReLU(),
#             nn.Linear(emb_size, self.num_experts),
#             nn.Softmax(dim=1)
#         )

#         # ---- Condition → Prompt Mapping ----
#         self.cond_to_prompt = nn.Linear(
#             Cond_OutDim,
#             self.prompt_len * self.prompt_dim
#         )

#         # =====================================================
#         # 5. Decoder
#         # =====================================================
#         Decoder_params = ["Channels"]
#         for key in Decoder.keys():
#             if key not in Decoder_params:
#                 raise Exception(f"{key} is not a param in Decoder")
#             if key == "Channels":
#                 Decoder_Channels = Decoder[key]
#                 assert len(Decoder_Channels) == 3

#         self.decoder = nn.Sequential(
#             nn.Linear(emb_size, Decoder_Channels[0]),
#             nn.ReLU(),
#             nn.BatchNorm1d(Decoder_Channels[0]),

#             nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
#             nn.ReLU(),

#             nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
#             nn.ReLU(),

#             nn.Linear(Decoder_Channels[2], 1)
#         )

#     # =====================================================
#     # Forward
#     # =====================================================
#     def forward(self, x, y=None, epoch=None, device=None, return_explain=False):

#         if device is None:
#             device = self.device

#         ReaPro_data, CatSol_data, RxnSmi, Cond_data = x

#         # -------- Graph --------
#         graph_emb = torch.cat([
#             self.ReaProEncoder(ReaPro_data),
#             self.CatSolEncoder(CatSol_data)
#         ], dim=1)

#         # -------- Text --------
#         text_emb = self.RxnSmiEncoder(RxnSmi)

#         # -------- Condition --------
#         cond_emb = self.CondEncoder(Cond_data)

#         # -------- Router --------
#         router_input = torch.cat([graph_emb, text_emb, cond_emb], dim=1)
#         expert_weights = self.router(router_input)

#         base_prompt = torch.einsum(
#             "bk,kld->bld",
#             expert_weights,
#             self.prompt_experts
#         )

#         cond_prompt = self.cond_to_prompt(cond_emb).view(
#             cond_emb.size(0),
#             self.prompt_len,
#             self.prompt_dim
#         )

#         dynamic_prompt = base_prompt + cond_prompt

#         graph_tokens = graph_emb.view(graph_emb.size(0), 2, self.prompt_dim)
#         graph_with_prompt = torch.cat([dynamic_prompt, graph_tokens], dim=1)

#         x = graph_with_prompt.mean(dim=1)
#         out = self.decoder(x)

#         if return_explain:
#             return out, {
#                 "expert_weights": expert_weights,
#                 "graph_emb": graph_emb,
#                 "text_emb": text_emb,
#                 "cond_emb": cond_emb,
#                 "dynamic_prompt": dynamic_prompt
#             }

#         return out


# Ablation experiments
class MMHRP_NoGraph_TextMain_MoPE(nn.Module):
    def __init__(self,
                 TextEncoder: dict,
                 ConditionEncoder: dict,
                 Decoder: dict,
                 emb_size: int = 128,
                 device: str = torch.device('cuda'),
                 FDS=None
                 ):
        super(MMHRP_NoGraph_TextMain_MoPE, self).__init__()

        self.device = device
        self.FDS = FDS
        self.emb_size = emb_size

        # =====================================================
        # 1. Text encoder (primary modality)
        # =====================================================
        Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
        for key in TextEncoder.keys():
            if key not in Text_params:
                raise Exception(f"{key} is not a param in Text Encoder")
            if key == "SmiFeatNum":
                self.SmiFeatNum = TextEncoder[key]
            if key == "Heads":
                Trans_Heads = TextEncoder[key]
            if key == "BigruChannels":
                BigruChannels = TextEncoder[key]
                bigru_input, bigru_hidden = BigruChannels
            if key == "BigruNumlayer":
                bigru_num_layers = TextEncoder[key]

        self.trans = nn.TransformerEncoderLayer(
            d_model=self.SmiFeatNum,
            nhead=Trans_Heads,
            batch_first=True,
            norm_first=True
        )

        self.bigru = BiGRU(
            bigru_input,
            bigru_hidden,
            bigru_num_layers,
            emb_size,
            device
        )

        self.RxnSmiEncoder = nn.Sequential(
            self.trans,
            self.bigru,
        )

        # =====================================================
        # 2. Numerical condition encoder
        # =====================================================
        Cond_params = ["InputDim", "HiddenDim", "OutDim"]
        for key in ConditionEncoder.keys():
            if key not in Cond_params:
                raise Exception(f"{key} is not a param in Condition Encoder")
            if key == "InputDim":
                Cond_InputDim = ConditionEncoder[key]
            if key == "HiddenDim":
                Cond_HiddenDim = ConditionEncoder[key]
            if key == "OutDim":
                Cond_OutDim = ConditionEncoder[key]

        self.CondEncoder = nn.Sequential(
            nn.Linear(Cond_InputDim, Cond_HiddenDim),
            nn.ReLU(),
            nn.Linear(Cond_HiddenDim, Cond_OutDim),
        )

        # =====================================================
        # 3. MoPE（Text Main）
        # =====================================================
        self.num_experts = 2
        self.prompt_len = 3
        self.prompt_dim = emb_size

        # ---- Prompt Experts ----
        self.prompt_experts = nn.Parameter(
            torch.randn(self.num_experts, self.prompt_len, self.prompt_dim)
        )

        # ---- Router（Text + Condition）----
        self.router = nn.Sequential(
            nn.Linear(emb_size + Cond_OutDim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, self.num_experts),
            nn.Softmax(dim=1)
        )

        # ---- Condition → Prompt Mapping ----
        self.cond_to_prompt = nn.Linear(
            Cond_OutDim,
            self.prompt_len * self.prompt_dim
        )

        # =====================================================
        # 4. Decoder
        # =====================================================
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception(f"{key} is not a param in Decoder")
            if key == "Channels":
                Decoder_Channels = Decoder[key]

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, Decoder_Channels[0]),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x, y=None, epoch=None, device=None):
        if device is None:
            device = self.device

        RxnSmi, Cond_data = x  # Graph modality removed

        # -------- Text --------
        text_emb = self.RxnSmiEncoder(RxnSmi)  # [B, emb]

        # -------- Condition --------
        cond_emb = self.CondEncoder(Cond_data)  # [B, Cond_OutDim]

        # =====================================================
        # MoPE Routing（Text + Condition）
        # =====================================================
        router_input = torch.cat([text_emb, cond_emb], dim=1)

        expert_weights = self.router(router_input)  # [B, k]

        base_prompt = torch.einsum(
            "bk,kld->bld",
            expert_weights,
            self.prompt_experts
        )

        cond_prompt = self.cond_to_prompt(cond_emb).view(
            cond_emb.size(0),
            self.prompt_len,
            self.prompt_dim
        )

        dynamic_prompt = base_prompt + cond_prompt  # [B, L, D]

        # =====================================================
        # Prompt injection into the text-primary representation
        # =====================================================
        text_tokens = text_emb.view(text_emb.size(0), 1, self.prompt_dim)

        text_with_prompt = torch.cat(
            [dynamic_prompt, text_tokens],
            dim=1
        )

        x = text_with_prompt.mean(dim=1)

        # -------- FDS --------
        if self.FDS is not None and y is not None and epoch is not None and epoch >= 1:
            self.FDS.cpu()
            x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
            y = y.to(device)

        # -------- Decoder --------
        out = self.decoder(x)
        return out


class MMHRP_GCL_NoText_GraphMain_MoPE(nn.Module):
    def __init__(self,
                 GraphEncoder: dict,
                 ConditionEncoder: dict,
                 Decoder: dict,
                 emb_size: int = 128,
                 device: str = torch.device('cuda'),
                 FDS=None
                 ):
        super(MMHRP_GCL_NoText_GraphMain_MoPE, self).__init__()

        self.device = device
        self.FDS = FDS
        self.emb_size = emb_size

        # =====================================================
        # 1. Graph encoder (primary modality, retained)
        # =====================================================
        Graph_params = ["NodeFeatNum", "Channels", "Heads"]
        for key in GraphEncoder.keys():
            if key not in Graph_params:
                raise Exception(f"{key} is not a param in Graph Encoder")
            if key == "NodeFeatNum":
                NodeFeatNum = GraphEncoder[key]
            if key == "Channels":
                GAT_Channels = GraphEncoder[key]
                assert len(GAT_Channels) == 2
            if key == "Heads":
                GAT_Heads = GraphEncoder[key]

        GATU_OutSize = sum([c * GAT_Heads for c in GAT_Channels])
        self.GATU_OutSize = GATU_OutSize * 2

        self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)

        self.ReaProEncoder = nn.Sequential(
            self.GATU_ReaPro,
            nn.Linear(self.GATU_OutSize, emb_size),
        )
        self.CatSolEncoder = nn.Sequential(
            self.GATU_CatSol,
            nn.Linear(self.GATU_OutSize, emb_size),
        )

        # =====================================================
        # 2. Condition encoder (retained)
        # =====================================================
        Cond_params = ["InputDim", "HiddenDim", "OutDim"]
        for key in ConditionEncoder.keys():
            if key not in Cond_params:
                raise Exception(f"{key} is not a param in Condition Encoder")
            if key == "InputDim":
                Cond_InputDim = ConditionEncoder[key]
            if key == "HiddenDim":
                Cond_HiddenDim = ConditionEncoder[key]
            if key == "OutDim":
                Cond_OutDim = ConditionEncoder[key]

        self.CondEncoder = nn.Sequential(
            nn.Linear(Cond_InputDim, Cond_HiddenDim),
            nn.ReLU(),
            nn.Linear(Cond_HiddenDim, Cond_OutDim),
        )

        # =====================================================
        # 3. MoPE（Graph-main Routing）
        # =====================================================
        self.num_experts = 2
        self.prompt_len = 3
        self.prompt_dim = emb_size

        # Prompt Experts
        self.prompt_experts = nn.Parameter(
            torch.randn(self.num_experts, self.prompt_len, self.prompt_dim)
        )

        # Router: Graph + Condition (text modality removed)
        self.router = nn.Sequential(
            nn.Linear(emb_size * 2 + Cond_OutDim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, self.num_experts),
            nn.Softmax(dim=1)
        )

        # Condition → Prompt
        self.cond_to_prompt = nn.Linear(
            Cond_OutDim,
            self.prompt_len * self.prompt_dim
        )

        # =====================================================
        # 4. Decoder (unchanged)
        # =====================================================
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception(f"{key} is not a param in Decoder")
            if key == "Channels":
                Decoder_Channels = Decoder[key]
                assert len(Decoder_Channels) == 3

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, Decoder_Channels[0]),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x, y=None, epoch=None, device=None):
        if device is None:
            device = self.device

        ReaPro_data, CatSol_data, Cond_data = x

        # -------- Graph --------
        graph_emb = torch.cat([
            self.ReaProEncoder(ReaPro_data),
            self.CatSolEncoder(CatSol_data)
        ], dim=1)  # [B, 2*emb]

        # -------- Condition --------
        cond_emb = self.CondEncoder(Cond_data)

        # =====================================================
        # MoPE Routing（Graph-main）
        # =====================================================
        router_input = torch.cat(
            [graph_emb, cond_emb],
            dim=1
        )

        expert_weights = self.router(router_input)

        base_prompt = torch.einsum(
            "bk,kld->bld",
            expert_weights,
            self.prompt_experts
        )

        cond_prompt = self.cond_to_prompt(cond_emb).view(
            cond_emb.size(0),
            self.prompt_len,
            self.prompt_dim
        )

        dynamic_prompt = base_prompt + cond_prompt

        # =====================================================
        # Prompt injection as graph virtual nodes
        # =====================================================
        graph_tokens = graph_emb.view(
            graph_emb.size(0), 2, self.prompt_dim
        )

        graph_with_prompt = torch.cat(
            [dynamic_prompt, graph_tokens],
            dim=1
        )

        x = graph_with_prompt.mean(dim=1)

        # -------- FDS --------
        if self.FDS is not None and y is not None and epoch is not None and epoch >= 1:
            self.FDS.cpu()
            x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
            y = y.to(device)

        # -------- Decoder --------
        out = self.decoder(x)
        return out

class MMHRP_GCL_tri_MoPE_NoCond_pdall(nn.Module):
    def __init__(self,
                 GraphEncoder: dict,
                 TextEncoder: dict,
                 Decoder: dict,
                 emb_size: int = 128,
                 device: str = torch.device('cuda'),
                 FDS=None
                 ):
        super(MMHRP_GCL_tri_MoPE_NoCond_pdall, self).__init__()

        self.device = device
        self.FDS = FDS
        self.emb_size = emb_size

        # =====================================================
        # 1. Graph encoder (primary modality)
        # =====================================================
        Graph_params = ["NodeFeatNum", "Channels", "Heads"]
        for key in GraphEncoder.keys():
            if key not in Graph_params:
                raise Exception(f"{key} is not a param in Graph Encoder")
            if key == "NodeFeatNum":
                NodeFeatNum = GraphEncoder[key]
            if key == "Channels":
                GAT_Channels = GraphEncoder[key]
                assert len(GAT_Channels) == 2
            if key == "Heads":
                GAT_Heads = GraphEncoder[key]

        GATU_OutSize = sum([c * GAT_Heads for c in GAT_Channels])
        self.GATU_OutSize = GATU_OutSize * 2

        self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)

        self.ReaProEncoder = nn.Sequential(
            self.GATU_ReaPro,
            nn.Linear(self.GATU_OutSize, emb_size),
        )
        self.CatSolEncoder = nn.Sequential(
            self.GATU_CatSol,
            nn.Linear(self.GATU_OutSize, emb_size),
        )

        # =====================================================
        # 2. Text Encoder（SMILES）
        # =====================================================
        Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
        for key in TextEncoder.keys():
            if key not in Text_params:
                raise Exception(f"{key} is not a param in Text Encoder")
            if key == "SmiFeatNum":
                self.SmiFeatNum = TextEncoder[key]
            if key == "Heads":
                Trans_Heads = TextEncoder[key]
            if key == "BigruChannels":
                BigruChannels = TextEncoder[key]
                bigru_input, bigru_hidden = BigruChannels
            if key == "BigruNumlayer":
                bigru_num_layers = TextEncoder[key]

        self.trans = nn.TransformerEncoderLayer(
            d_model=self.SmiFeatNum,
            nhead=Trans_Heads,
            batch_first=True,
            norm_first=True
        )
        self.bigru = BiGRU(
            bigru_input,
            bigru_hidden,
            bigru_num_layers,
            emb_size,
            device
        )

        self.RxnSmiEncoder = nn.Sequential(
            self.trans,
            self.bigru,
        )

        # =====================================================
        # 3. MoPE without numerical conditions
        # =====================================================
        self.num_experts = 2
        self.prompt_len = 3
        self.prompt_dim = emb_size

        # ---- Prompt Experts ----
        self.prompt_experts = nn.Parameter(
            torch.randn(self.num_experts, self.prompt_len, self.prompt_dim)
        )

        # ---- Router（Graph + Text ONLY）----
        self.router = nn.Sequential(
            nn.Linear(emb_size * 3, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, self.num_experts),
            nn.Softmax(dim=1)
        )

        # =====================================================
        # 4. Decoder
        # =====================================================
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception(f"{key} is not a param in Decoder")
            if key == "Channels":
                Decoder_Channels = Decoder[key]
                assert len(Decoder_Channels) == 3

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, Decoder_Channels[0]),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x, y=None, epoch=None, device=None):
        if device is None:
            device = self.device

        ReaPro_data, CatSol_data, RxnSmi = x  # Numerical conditions removed

        # -------- Graph --------
        graph_emb = torch.cat([
            self.ReaProEncoder(ReaPro_data),
            self.CatSolEncoder(CatSol_data)
        ], dim=1)  # [B, 2*emb]

        # -------- Text --------
        text_emb = self.RxnSmiEncoder(RxnSmi)  # [B, emb]

        # =====================================================
        # MoPE Routing（No Condition）
        # =====================================================
        router_input = torch.cat(
            [graph_emb, text_emb],
            dim=1
        )

        expert_weights = self.router(router_input)  # [B, k]

        base_prompt = torch.einsum(
            "bk,kld->bld",
            expert_weights,
            self.prompt_experts
        )

        # =====================================================
        # Prompt injection into the graph-primary representation
        # =====================================================
        graph_tokens = graph_emb.view(
            graph_emb.size(0), 2, self.prompt_dim
        )

        graph_with_prompt = torch.cat(
            [base_prompt, graph_tokens],
            dim=1
        )

        x = graph_with_prompt.mean(dim=1)

        # -------- FDS --------
        if self.FDS is not None and y is not None and epoch is not None and epoch >= 1:
            self.FDS.cpu()
            x = self.FDS.smooth(x.cpu(), y.cpu(), epoch).to(device)
            y = y.to(device)

        # -------- Decoder --------
        out = self.decoder(x)
        return out



class MMHRP_GCL_tri_NoMoPE_pdall(nn.Module):
    """
    Ablation model:
    - Remove MoPE (Prompt-level Experts & Routing)
    - Graph modality remains the main modality
    - Text & Condition are fused by static concatenation
    """

    def __init__(self,
                 GraphEncoder: dict,
                 TextEncoder: dict,
                 ConditionEncoder: dict,
                 Decoder: dict,
                 emb_size: int = 128,
                 device: str = torch.device('cuda'),
                 FDS=None
                 ):
        super(MMHRP_GCL_tri_NoMoPE_pdall, self).__init__()

        self.device = device
        self.FDS = FDS
        self.emb_size = emb_size

        # =====================================================
        # 1. Graph encoder (primary modality)
        # =====================================================
        Graph_params = ["NodeFeatNum", "Channels", "Heads"]
        for key in GraphEncoder.keys():
            if key not in Graph_params:
                raise Exception(f"{key} is not a param in Graph Encoder")
            if key == "NodeFeatNum":
                NodeFeatNum = GraphEncoder[key]
            if key == "Channels":
                GAT_Channels = GraphEncoder[key]
                assert len(GAT_Channels) == 2
            if key == "Heads":
                GAT_Heads = GraphEncoder[key]

        GATU_OutSize = sum([c * GAT_Heads for c in GAT_Channels])
        self.GATU_OutSize = GATU_OutSize * 2

        self.GATU_ReaPro = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)
        self.GATU_CatSol = GATU(NodeFeatNum, GAT_Channels, GAT_Heads)

        self.ReaProEncoder = nn.Sequential(
            self.GATU_ReaPro,
            nn.Linear(self.GATU_OutSize, emb_size),
        )
        self.CatSolEncoder = nn.Sequential(
            self.GATU_CatSol,
            nn.Linear(self.GATU_OutSize, emb_size),
        )

        # =====================================================
        # 2. Text Encoder（SMILES）
        # =====================================================
        Text_params = ["SmiFeatNum", "Heads", "BigruChannels", "BigruNumlayer"]
        for key in TextEncoder.keys():
            if key not in Text_params:
                raise Exception(f"{key} is not a param in Text Encoder")
            if key == "SmiFeatNum":
                self.SmiFeatNum = TextEncoder[key]
            if key == "Heads":
                Trans_Heads = TextEncoder[key]
            if key == "BigruChannels":
                BigruChannels = TextEncoder[key]
                bigru_input, bigru_hidden = BigruChannels
            if key == "BigruNumlayer":
                bigru_num_layers = TextEncoder[key]

        self.trans = nn.TransformerEncoderLayer(
            d_model=self.SmiFeatNum,
            nhead=Trans_Heads,
            batch_first=True,
            norm_first=True
        )

        self.bigru = BiGRU(
            bigru_input,
            bigru_hidden,
            bigru_num_layers,
            emb_size,
            device
        )

        self.RxnSmiEncoder = nn.Sequential(
            self.trans,
            self.bigru,
        )

        # =====================================================
        # 3. Numerical condition encoder
        # =====================================================
        Cond_params = ["InputDim", "HiddenDim", "OutDim"]
        for key in ConditionEncoder.keys():
            if key not in Cond_params:
                raise Exception(f"{key} is not a param in Condition Encoder")
            if key == "InputDim":
                Cond_InputDim = ConditionEncoder[key]
            if key == "HiddenDim":
                Cond_HiddenDim = ConditionEncoder[key]
            if key == "OutDim":
                Cond_OutDim = ConditionEncoder[key]

        self.CondEncoder = nn.Sequential(
            nn.Linear(Cond_InputDim, Cond_HiddenDim),
            nn.ReLU(),
            nn.Linear(Cond_HiddenDim, Cond_OutDim),
        )

        # =====================================================
        # 4. Static multimodal fusion without MoPE
        # =====================================================
        fusion_dim = emb_size * 2 + emb_size + Cond_OutDim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, emb_size),
            nn.ReLU()
        )

        # =====================================================
        # 5. Decoder
        # =====================================================
        Decoder_params = ["Channels"]
        for key in Decoder.keys():
            if key not in Decoder_params:
                raise Exception(f"{key} is not a param in Decoder")
            if key == "Channels":
                Decoder_Channels = Decoder[key]
                assert len(Decoder_Channels) == 3

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, Decoder_Channels[0]),
            nn.ReLU(),
            nn.BatchNorm1d(Decoder_Channels[0]),

            nn.Linear(Decoder_Channels[0], Decoder_Channels[1]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[1], Decoder_Channels[2]),
            nn.ReLU(),

            nn.Linear(Decoder_Channels[2], 1)
        )

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x, y=None, epoch=None, device=None):
        if device is None:
            device = self.device

        ReaPro_data, CatSol_data, RxnSmi, Cond_data = x

        # -------- Graph (Main modality) --------
        graph_emb = torch.cat([
            self.ReaProEncoder(ReaPro_data),
            self.CatSolEncoder(CatSol_data)
        ], dim=1)  # [B, 2*emb]

        # -------- Text --------
        text_emb = self.RxnSmiEncoder(RxnSmi)  # [B, emb]

        # -------- Condition --------
        cond_emb = self.CondEncoder(Cond_data)  # [B, Cond_OutDim]

        # -------- Static Fusion --------
        fused = torch.cat(
            [graph_emb, text_emb, cond_emb],
            dim=1
        )
        fused = self.fusion_layer(fused)

        # -------- FDS --------
        if self.FDS is not None and y is not None and epoch is not None and epoch >= 1:
            self.FDS.cpu()
            fused = self.FDS.smooth(fused.cpu(), y.cpu(), epoch).to(device)
            y = y.to(device)

        # -------- Decoder --------
        out = self.decoder(fused)
        return out

# Model Evaluation Function


def RMSE(pred, true):
    return np.sqrt(mean_squared_error(true, pred))


def R2(pred, true):
    return r2_score(true, pred)


def MAE(pred, true):
    return mean_absolute_error(true, pred)
