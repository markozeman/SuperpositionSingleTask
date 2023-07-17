import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear, Dropout, LayerNorm, MultiheadAttention
from torch.nn import functional as F


class Transformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes):
        super(Transformer, self).__init__()

        transformer_encoder_layer = TransformerEncoderLayer(input_size, num_heads, dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 256, input_size),   # 8192 -> 32
            nn.ReLU(),
            nn.Linear(input_size, num_classes),  # 32 -> 2
        )

        # self.init_weights()

    # def init_weights(self):
    #     initrange = 1e-10
    #     self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, key_padding_mask):
        x = self.transformer_encoder(input, src_key_padding_mask=key_padding_mask)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = self.mlp(x)   # feed through MLP
        return output


class MyTransformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes, use_PSP=False):
        super(MyTransformer, self).__init__()

        if use_PSP:
            # transformer_encoder_layer = MyTransformerEncoderLayerPSP(input_size, num_heads, dim_feedforward, batch_first=True)
            self.transformer_encoder = MyTransformerEncoderLayerPSP(input_size, num_heads, dim_feedforward, batch_first=True)
            # self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)

            self.linear_1 = nn.Linear(input_size * 256, input_size)
            self.linear_2 = nn.Linear(input_size, num_classes)
            self.trainable_layers = [self.linear_1, self.linear_2]

        else:
            transformer_encoder_layer = MyTransformerEncoderLayer(input_size, num_heads, dim_feedforward, batch_first=True)
            self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)

            self.mlp = nn.Sequential(
                nn.Linear(input_size * 256, input_size),   # 8192 -> 32
                nn.ReLU(),
                nn.Linear(input_size, num_classes),  # 32 -> 2
            )

    def forward(self, input, key_padding_mask, use_PSP=False, contexts=None, task_id=None):
        if use_PSP:
            x = self.transformer_encoder(input, None, src_key_padding_mask=key_padding_mask, contexts=contexts, task_id=task_id)
            x = torch.flatten(x, start_dim=1, end_dim=2)

            for i, lyr in enumerate(self.trainable_layers):
                context_matrix = torch.from_numpy(np.diag(contexts[task_id][i+4]).astype(np.float32)).cuda()    # i+4, because we already had 4 layers in transformer encoder
                x = torch.matmul(x, context_matrix)
                x = lyr(x)
                if i < len(self.trainable_layers) - 1:  # apply ReLU if it is not the last layer
                    x = nn.functional.relu(x)
            output = x

        else:
            x = self.transformer_encoder(input, mask=None, src_key_padding_mask=key_padding_mask)
            x = torch.flatten(x, start_dim=1, end_dim=2)
            output = self.mlp(x)   # feed through MLP
        return output


class MyTransformerEncoderLayer(nn.Module):

    def __init__(self, input_size, num_heads, dim_feedforward, batch_first, dropout=0.0, activation=F.relu):
        super(MyTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=batch_first)

        # implementation of feed-forward model
        self.linear1 = Linear(input_size, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, input_size)

        # # elementwise_affine=False means no trainable parameters (if True, there are 64 trainable parameters)
        # self.norm1 = LayerNorm(input_size, elementwise_affine=False)
        # self.norm2 = LayerNorm(input_size, elementwise_affine=False)

        self.layer_norm = LayerNorm(input_size, elementwise_affine=False)

        self.activation = activation

    def self_attention_block(self, input, key_padding_mask):
        x = self.self_attn(input, input, input, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, input):
        x = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return self.dropout(x)

    def forward(self, input, src_mask, src_key_padding_mask):
        x = self.layer_norm(input + self.self_attention_block(input, src_key_padding_mask))   # self.norm1
        x = self.layer_norm(x + self.ff_block(x))   # self.norm2
        return x


class MyTransformerEncoderLayerPSP(nn.Module):

    def __init__(self, input_size, num_heads, dim_feedforward, batch_first, dropout=0.0, activation=F.relu):
        super(MyTransformerEncoderLayerPSP, self).__init__()

        self.input_size = input_size

        self.self_attn = MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=batch_first)

        # implementation of feed-forward model
        self.linear1 = Linear(input_size, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, input_size)

        self.layer_norm = LayerNorm(input_size, elementwise_affine=False)

        self.activation = activation

    def self_attention_block(self, input, key_padding_mask, contexts=None, task_id=None):
        context_matrix = torch.from_numpy(np.diag(contexts[task_id][1][:self.input_size]).astype(np.float32)).cuda()
        x = self.self_attn(torch.matmul(input, context_matrix), torch.matmul(input, context_matrix), torch.matmul(input, context_matrix),
                           key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, input, contexts=None, task_id=None):
        context_matrix1 = torch.from_numpy(np.diag(contexts[task_id][2]).astype(np.float32)).cuda()
        context_matrix2 = torch.from_numpy(np.diag(contexts[task_id][3]).astype(np.float32)).cuda()
        x = self.linear2(torch.matmul(self.dropout(self.activation(self.linear1(torch.matmul(input, context_matrix1)))), context_matrix2))
        return self.dropout(x)

    def forward(self, input, src_mask, src_key_padding_mask, contexts=None, task_id=None):
        context_matrix = torch.from_numpy(np.diag(contexts[task_id][0][:self.input_size]).astype(np.float32)).cuda()
        x = self.layer_norm(torch.matmul(input, context_matrix) + self.self_attention_block(input, src_key_padding_mask, contexts, task_id))   # self.norm1
        x = self.layer_norm(x + self.ff_block(x, contexts, task_id))   # self.norm2
        return x


class MLP(nn.Module):

    def __init__(self, input_size, num_classes, use_PSP=False):
        super(MLP, self).__init__()

        # todo
        # hidden_size = 41    # 41 - to match (or slightly increase) number of parameters in transformer model
        hidden_size = 100

        if use_PSP:
            self.linear_1 = nn.Linear(input_size * 256, hidden_size)
            self.linear_2 = nn.Linear(hidden_size, num_classes)
            self.trainable_layers = [self.linear_1, self.linear_2]
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_size * 256, hidden_size),   # 8192 -> 41
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),  # 41 -> 2
            )

    def forward(self, input, use_PSP=False, contexts=None, task_id=None):
        if input.ndim > 2:
            x = torch.flatten(input, start_dim=1, end_dim=2)
        else:
            x = input

        if use_PSP:
            for i, lyr in enumerate(self.trainable_layers):
                context_matrix = torch.from_numpy(np.diag(contexts[task_id][i]).astype(np.float32)).cuda()
                x = torch.matmul(x, context_matrix)
                x = lyr(x)
                if i < len(self.trainable_layers) - 1:  # apply ReLU if it is not the last layer
                    x = nn.functional.relu(x)
            output = x
        else:
            output = self.mlp(x)

        return output


class AdapterTransformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes, bottleneck_size=16):
        super(AdapterTransformer, self).__init__()

        adapter_transformer_encoder_layer = AdapterTransformerEncoderLayer(input_size, num_heads, dim_feedforward,
                                                                           batch_first=True, bottleneck_size=bottleneck_size)
        self.transformer_encoder = TransformerEncoder(adapter_transformer_encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 256, input_size),   # 8192 -> 32
            nn.ReLU(),
            nn.Linear(input_size, num_classes),  # 32 -> 2
        )

    def forward(self, input, key_padding_mask):
        x = self.transformer_encoder(input, mask=None, src_key_padding_mask=key_padding_mask)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = self.mlp(x)   # feed through MLP
        return output


class AdapterTransformerEncoderLayer(nn.Module):

    def __init__(self, input_size, num_heads, dim_feedforward, batch_first, dropout=0.0, activation=F.relu, bottleneck_size=16):
        super(AdapterTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=batch_first)

        # implementation of feed-forward model
        self.linear1 = Linear(input_size, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, input_size)

        # # elementwise_affine=False means no trainable parameters (if True, there are 64 trainable parameters)
        # self.norm1 = LayerNorm(input_size, elementwise_affine=False)
        # self.norm2 = LayerNorm(input_size, elementwise_affine=False)

        self.layer_norm = LayerNorm(input_size, elementwise_affine=False)
        self.layer_norm_trainable = LayerNorm(input_size, elementwise_affine=True)

        self.activation = activation

        self.adapter_layer = AdapterLayer(input_size, bottleneck_size)
        # self.adapter_layer = AdapterLayer1to1(input_size)

    def self_attention_block(self, input, key_padding_mask):
        x = self.self_attn(input, input, input, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, input):
        x = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return self.dropout(x)

    def forward(self, input, src_mask, src_key_padding_mask):
        x = self.layer_norm(input + self.self_attention_block(input, src_key_padding_mask))  # self.norm1
        x = self.layer_norm(x + self.ff_block(x))  # self.norm2
        x = self.layer_norm_trainable(input + self.adapter_layer(x))    # adapter
        return x


class AdapterLayer(nn.Module):

    def __init__(self, input_size, bottleneck_size=16):
        super(AdapterLayer, self).__init__()

        self.bottleneck_mlp = nn.Sequential(
            nn.Linear(input_size, bottleneck_size),  # 32 -> bottleneck_size
            nn.ReLU(),
            nn.Linear(bottleneck_size, input_size),  # bottleneck_size -> 32
        )

    def forward(self, input):
        x = self.bottleneck_mlp(input)
        output = x + input
        return output


class AdapterLayer1to1(nn.Module):

    def __init__(self, input_size):
        super(AdapterLayer1to1, self).__init__()

        self.one2one = nn.Linear(input_size, input_size)
        self.one2one.weight = torch.nn.Parameter(torch.randn(input_size, 1, requires_grad=True), requires_grad=True)
        self.one2one.bias = torch.nn.Parameter(torch.zeros(input_size, requires_grad=False), requires_grad=False)   # bias not used

    def forward(self, input):
        return self.one2one(input)


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()

        hidden_size = 100

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.layer(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()

        # hidden_size = 10
        #
        # self.layer = nn.Sequential(
        #     nn.Linear(input_dim, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_experts),
        #     nn.Softmax(dim=1)
        # )

        self.layer = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, num_classes, num_gates):
        super(MixtureOfExperts, self).__init__()
        self.expert = Expert(input_dim, num_classes)
        self.gating_network = GatingNetwork(input_dim, num_gates)
        self.num_gates = num_gates
        self.num_classes_per_gate = num_classes // num_gates

    def forward(self, x):
        weights = self.gating_network(x)  # weights for each group of classes
        outputs = self.expert(x)  # outputs from the expert

        # Apply gating weights to expert outputs
        outputs_weighted = torch.zeros(outputs.shape).to(outputs.device)
        for i in range(self.num_gates):
            outputs_weighted[:, i * self.num_classes_per_gate:(i + 1) * self.num_classes_per_gate] = \
                outputs[:, i * self.num_classes_per_gate:(i + 1) * self.num_classes_per_gate] * weights[:, i].unsqueeze(-1)

        return outputs_weighted  # weighted output of expert


