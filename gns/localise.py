import torch.nn as nn
import torch
import math
import rff
from gns.geometry import construct_3d_basis_from_2_vectors, multiply_matrices
from gns.my_egnn import EGNN_rot


class FrameMLP(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        hidden_dim=256,
        num_features=6,
        trajectory_size=5,
        dropout=0.0,
        n_layers=4,
        n_out_dims=6,
        particle_type_embedding_size=16,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            n_out_dims - Output of MLP is 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.trajectory_size = trajectory_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_out_dims = n_out_dims
        self.input_dims = 2 * (self.trajectory_size + 1) + particle_type_embedding_size
        # Layers/Networks
        # Define an MLP to pass the transformed features through num_layer times and use GeLU as the activation function
        self.mlp = nn.Sequential()

        # Input layer
        self.mlp.add_module(
            "input_layer",
            # rff.layers.GaussianEncoding(
            #     sigma=10.0,
            #     input_size=2 * self.trajectory_size,
            #     encoded_size=self.embed_dim,
            # ),
            nn.Linear(self.input_dims, self.embed_dim),
        )
        self.mlp.add_module("input_layer_activation", nn.GELU())
        self.mlp.add_module("input_layer_dropout", nn.Dropout(self.dropout))
        # Embedding layer
        self.mlp.add_module(
            "hidden_layer_0",
            nn.Linear(self.embed_dim, self.hidden_dim),
        )
        self.mlp.add_module("hidden_layer_0_activation", nn.GELU())
        self.mlp.add_module("hidden_layer_0_dropout", nn.Dropout(self.dropout))
        # Hidden layers
        for i in range(1, self.n_layers - 1):
            self.mlp.add_module(
                "hidden_layer_{}".format(i), nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.mlp.add_module("hidden_layer_{}_activation".format(i), nn.GELU())
            self.mlp.add_module(
                "hidden_layer_{}_dropout".format(i), nn.Dropout(self.dropout)
            )
        # Output layer
        self.mlp.add_module(
            "output_layer", nn.Linear(self.hidden_dim, 2 * (self.trajectory_size + 1))
        )

        self.gvp_linear = nn.Linear(2 * (self.trajectory_size + 1), 2, bias=False)

    def forward(self, x, node_embeddings):
        # Preprocess input
        # The input is of shape [n_particles, (n_timesteps+1)*n_features] where n_features is 4 for 2D coordinates
        # and 6 for 3D coordinates. The first n_timesteps are the positions and velocities of the particle
        # and the last 2 features are the distances (upper, lower) from the boundary.
        norm_x = x.reshape(-1, 2, 3).norm(dim=-1)
        norm_x = norm_x.reshape(-1, (self.trajectory_size + 1) * 2)
        mlp_input = torch.cat([norm_x, node_embeddings], dim=-1)
        out = self.mlp(mlp_input).unsqueeze(-1)

        x = x.reshape(-1, (self.trajectory_size + 1) * 2, 3)

        # BN,2*T,3
        y = out * x
        y = self.gvp_linear(y.transpose(-1, -2)).transpose(-1, -2)
        # Now y is BN,2T,3
        # The output is a 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
        # we can construct an SO(3) rotation matrix and a translation vector.
        v1, v2 = y[..., 0, :], y[..., 1, :]

        R = construct_3d_basis_from_2_vectors(v1, v2)

        return R


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class FrameTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        hidden_dim=256,
        num_features=6,
        trajectory_size=5,
        dropout=0.0,
        n_layers=4,
        num_heads=8,
        n_out_dims=6,
        particle_type_embedding_size=16,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.trajectory_size = trajectory_size
        self.dropout = dropout
        self.num_layers = n_layers
        self.n_out_dims = n_out_dims
        self.input_dims = 2 * (self.trajectory_size) + particle_type_embedding_size
        self.num_heads = num_heads
        self.particle_type_embedding_size = particle_type_embedding_size
        self.embed_norms_dim = embed_dim - self.particle_type_embedding_size
        # Layers/Networks
        self.embed_norms = rff.layers.GaussianEncoding(
            sigma=5.0, input_size=2, encoded_size=self.embed_norms_dim // 2
        )
        # self.input_layer = nn.Linear(self.embed_dim, self.embed_dim)

        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    self.embed_dim,
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 2 * (self.trajectory_size + 1)),
        )
        self.dropout = nn.Dropout(self.dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.positional_encoding = PositionalEncoding(
            d_model=self.embed_dim, max_len=1 + self.trajectory_size
        )

        self.gvp_linear = nn.Linear(2 * (self.trajectory_size + 1), 2, bias=False)

    def forward(self, x, node_embeddings):
        # Preprocess input
        # Split the tensor into two parts
        nparticles = x.shape[0]
        v_tensor, p_tensor = torch.chunk(x[..., :-6], 2, dim=-1)
        v_norm = v_tensor.reshape(nparticles, self.trajectory_size, 3).norm(dim=-1)
        p_norm = p_tensor.reshape(nparticles, self.trajectory_size, 3).norm(dim=-1)

        # Zip the tensors together
        norm_x = torch.stack((v_norm, p_norm), dim=-1)
        # norm_x = norm_x.reshape(-1, (self.trajectory_size), 2)
        B2, _, _ = norm_x.shape
        norm_x = self.embed_norms(norm_x)

        # The shape of norm_x is now [B*N, T, embed_dim]
        # Add node embeddings to input so for every T we have the same node embeddings
        # The node embeddings are of shape [B*N, 16] it needs to be repeated T times
        n_embed = (
            node_embeddings.repeat(1, self.trajectory_size)
            .reshape(-1, self.trajectory_size, self.particle_type_embedding_size)
            .to(norm_x.device)
        )
        transformer_input = torch.cat([norm_x, n_embed], dim=-1).to(norm_x.device)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B2, 1, 1)
        transformer_input = torch.cat([cls_token, transformer_input], dim=1)
        transformer_input = self.positional_encoding(transformer_input)
        # Apply Transforrmer
        transformer_input = self.dropout(transformer_input)
        transformer_input = transformer_input.transpose(0, 1)
        transformer_output = self.transformer(transformer_input)

        # Perform classification prediction
        cls = transformer_output[0]
        out = self.mlp_head(cls).unsqueeze(-1)

        # Preprocess input
        # The input is of shape [n_particles, (n_timesteps+1)*n_features] where n_features is 4 for 2D coordinates
        # and 6 for 3D coordinates. The first n_timesteps are the positions and velocities of the particle
        # and the last 2 features are the distances (upper, lower) from the boundary.

        x = x.reshape(-1, (self.trajectory_size + 1) * 2, 3)

        # BN,2*T,3
        y = out * x
        y = self.gvp_linear(y.transpose(-1, -2)).transpose(-1, -2)
        # Now y is BN,2T,3
        # The output is a 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
        # we can construct an SO(3) rotation matrix and a translation vector.
        v1, v2 = y[..., 0, :], y[..., 1, :]

        R = construct_3d_basis_from_2_vectors(v1, v2)

        return R


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        hidden_dim=256,
        num_features=6,
        trajectory_size=5,
        dropout=0.0,
        n_layers=4,
        num_heads=8,
        n_out_dims=6,
        particle_type_embedding_size=16,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.trajectory_size = trajectory_size
        self.dropout = dropout
        self.num_layers = n_layers
        self.n_out_dims = n_out_dims
        self.input_dims = 2 * (self.trajectory_size) + particle_type_embedding_size
        self.num_heads = num_heads
        self.particle_type_embedding_size = particle_type_embedding_size
        self.embed_norms_dim = embed_dim - self.particle_type_embedding_size
        # Layers/Networks
        self.embed_norms = rff.layers.GaussianEncoding(
            sigma=5.0, input_size=2, encoded_size=self.embed_norms_dim // 2
        )
        # self.input_layer = nn.Linear(self.embed_dim, self.embed_dim)

        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    self.embed_dim,
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.dropout = nn.Dropout(self.dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.positional_encoding = PositionalEncoding(
            d_model=self.embed_dim, max_len=1 + self.trajectory_size
        )

    def forward(self, x, node_embeddings):
        # Preprocess input
        # Split the tensor into two parts
        v_tensor, p_tensor = torch.chunk(x[..., :-6], 2, dim=-1)
        v_norm = v_tensor.reshape(-1, self.trajectory_size, 3).norm(dim=-1)
        p_norm = p_tensor.reshape(-1, self.trajectory_size, 3).norm(dim=-1)

        # Zip the tensors together
        norm_x = torch.stack((v_norm, p_norm), dim=-1)
        B2, _, _ = norm_x.shape
        norm_x = self.embed_norms(norm_x)

        # The shape of norm_x is now [B*N, T, embed_dim]
        # Add node embeddings to input so for every T we have the same node embeddings
        # The node embeddings are of shape [B*N, 16] it needs to be repeated T times
        n_embed = (
            node_embeddings.repeat(1, self.trajectory_size)
            .reshape(-1, self.trajectory_size, self.particle_type_embedding_size)
            .to(norm_x.device)
        )
        transformer_input = torch.cat([norm_x, n_embed], dim=-1).to(norm_x.device)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B2, 1, 1)
        transformer_input = torch.cat([cls_token, transformer_input], dim=1)
        transformer_input = self.positional_encoding(transformer_input)
        # Apply Transforrmer
        transformer_input = self.dropout(transformer_input)
        transformer_input = transformer_input.transpose(0, 1)
        transformer_output = self.transformer(transformer_input)

        # Perform classification prediction
        cls = transformer_output[0]
        out = self.mlp_head(cls)

        return out


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        params,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.embed_dim = params.get("localizer_embedding_size", 128)
        self.hidden_dim = params.get("localizer_hidden_size", 256)
        self.num_features = params.get("localizer_n_in_dims", 6)
        self.n_out_dims = params.get("localizer_n_out_dims", 6)
        self.num_heads = params.get("localizer_n_heads", 4)
        self.num_layers = params.get("localizer_n_layers", 2)
        self.trajectory_size = params.get("window_size", 50)
        self.dropout_prob = params.get("localizer_dropout", 0.0)
        self.num_objects = params.get("localizer_n_objects", 5)
        # Layers/Networks
        self.input_layer = nn.Linear(2 * self.trajectory_size, self.embed_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    self.embed_dim,
                    self.hidden_dim,
                    self.num_heads,
                    dropout=self.dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        # Preprocess input
        B, T, N, F = x.shape
        # Charge is also a feature
        norm_x = x.unflatten(-1, (2, 3)).norm(dim=-1)
        norm_x = norm_x.permute(0, 2, 1, 3).reshape(B, N, T * 2)
        norm_x = self.input_layer(norm_x)
        # norm_x = norm_x + self.pos_embedding
        # Apply Transforrmer
        norm_x = self.dropout(norm_x)
        norm_x = norm_x.transpose(0, 1)
        norm_x = self.transformer(norm_x)

        # Perform classification prediction
        # B,N,E
        out = self.mlp_head(norm_x).transpose(0, 1)
        return out


class SpatioTemporalEGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        recurrent=False,
        norm_diff=False,
        tanh=False,
    ):
        super().__init__()
        self.EGNN = EGNN_rot(
            in_node_nf,
            in_edge_nf,
            hidden_nf,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            coords_weight=coords_weight,
            recurrent=recurrent,
            norm_diff=norm_diff,
            tanh=tanh,
        )

        self.device = device

    def forward(self, h, edges, rot_rep, edge_attr):
        x, h = self.EGNN(h, rot_rep, edges, edge_attr)
        return h, x.reshape(-1, 3, 2)


class SpatioTemporalFrame(nn.Module):
    """
    Module that takes as input the node features and outputs a rotation matrix.
    The procedure goes as follows:
        1. Create temporal embedding for each node of the batch using a transformer encoder (rotation invariant)
        2. Create spatial embedding for each node of the batch using a transformer encoder (rotation invariant)
        3. Using the spatiotemporal embedding, perform cross attention between the spatial and temporal embedding
           to produce a two scalar values for each node of the batch [B,N,2] where H is the hidden dimension,
           to multiply with the vector features of each node and create a [B,N,2,3] result.
        4. Using a GVP layer (linear layer without bias) of the form [B,N,2*T,3] -> [B,N,2,3] we create
           two 3D vectors which using Gram-Scmidt orthogonalization we can construct an SO(3) rotation matrix.
    Inputs:
            embed_dim - Dimensionality of the input feature vectors
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
            num_features - Number of features each node has (e.g. 4 for 2D coordinates [p_x,p_y,v_x,v_y]
                           and 6 for 3D coordinates [p_x,p_y,p_z,v_x,v_y,v_z]])
            n_out_dims - Output of MLP is 6D representation of two vectors, which by using Gram-Schmidt orthogonalization
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
    """

    def __init__(
        self,
        embed_dim=128,
        hidden_dim=256,
        num_features=6,
        trajectory_size=5,
        dropout=0.0,
        n_layers=4,
        num_heads=8,
        n_out_dims=6,
        particle_type_embedding_size=16,
        device="cpu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.trajectory_size = trajectory_size
        self.dropout = dropout
        self.n_out_dims = 2 * self.trajectory_size

        self.temporal_encoder = TemporalEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_features=num_features,
            trajectory_size=trajectory_size,
            dropout=dropout,
            n_layers=n_layers,
            num_heads=num_heads,
            n_out_dims=n_out_dims,
            particle_type_embedding_size=particle_type_embedding_size,
        )

        self.spatial_encoder = SpatioTemporalEGNN(
            in_node_nf=embed_dim,
            in_edge_nf=1,
            hidden_nf=hidden_dim,
            act_fn=nn.SiLU(),
            n_layers=10,
            coords_weight=1.0,
            recurrent=False,
            norm_diff=False,
            tanh=False,
            device=device,
        )

        # self.gvp_linear = nn.Linear(2 * self.trajectory_size, 2, bias=False)

    def forward(self, x, particle_type_embeddings, edges, edge_attr):
        # Rotation representation is the velocity vector of last time step and the difference between the last two time steps
        v_tensor, p_tensor = torch.chunk(x[..., :-6], 2, dim=-1)
        v_tensor = v_tensor.reshape(-1, self.trajectory_size, 3)
        # # Acceleration tensor is the difference between the last two time steps
        a_tensor = v_tensor[:, -1] - v_tensor[:, -2]
        # Concatenate the last velocity and acceleration tensors such that the shape is [BN,3,2]
        rot_vec = (
            torch.cat([v_tensor[:, -1], a_tensor], dim=-1)
            .reshape(-1, 2, 3)
            .transpose(-1, -2)
        )
        # rot_vec = x[..., -6:].reshape(-1, 3, 2)
        # Temporal embedding
        temporal_embed = self.temporal_encoder(x, particle_type_embeddings)

        # Spatial embedding
        spatiotemporal_embed, v = self.spatial_encoder(
            temporal_embed, edges, rot_vec, edge_attr
        )
        # Spatial-temporal embedding
        # st_embed = self.cross_attention(spatial_embed, temporal_embed)

        # Construct the rotation matrix
        R = construct_3d_basis_from_2_vectors(v[..., 0], v[..., 1])

        return R
