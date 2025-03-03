import torch
import torch.nn as nn
import math

class GlobalFeatureEmbedding(nn.Module):
    """Embeds global features (Mach, Reynolds, z-coordinate) into the model dimension"""
    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)
        
    def forward(self, mach, reynolds, z_coord):
        global_features = torch.stack([mach, reynolds, z_coord], dim=-1)
        return self.embedding(global_features)

class SplineFeatureEmbedding(nn.Module):
    """Projects spline features into the model dimension"""
    def __init__(self, d_model, include_pressure=True):
        super().__init__()
        self.input_dim = 4 if include_pressure else 3  # 3 features: arc_length, x, y (optionally pressure)
        self.embedding = nn.Linear(self.input_dim, d_model)
        
    def forward(self, spline_features):
        return self.embedding(spline_features)

class AirfoilTransformerDecoder(nn.Module):
    def __init__(
        self,
        enable_transfer_learning,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        
        if enable_transfer_learning:
            # Embeddings for source (2D) features including pressure
            self.source_embedding = SplineFeatureEmbedding(d_model, include_pressure=True)
            # Embeddings for target (3D) geometric features only
            self.target_embedding = SplineFeatureEmbedding(d_model, include_pressure=False)
            self.global_embedding = GlobalFeatureEmbedding(d_model)
        elif not enable_transfer_learning:
            # Embeddings for source (2D) features including pressure
            self.source_embedding = SplineFeatureEmbedding(d_model, include_pressure=False)
            # Embeddings for target (3D) geometric features only
            self.target_embedding = SplineFeatureEmbedding(d_model, include_pressure=False)
            self.global_embedding = GlobalFeatureEmbedding(d_model)
        else:
            sys.exit('Transfer learning type must be explicitly defined')
        
        # Create decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Stack decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection to single pressure value
        self.output_projection = nn.Linear(d_model, 1)  # Project to pressure only
        
    def forward(
        self,
        src_features,      # [batch_size, src_seq_len, 4] - 2D airfoil with pressure
        tgt_geometry,      # [batch_size, tgt_seq_len, 3] - 3D geometry without pressure
        mach,              # [batch_size]
        reynolds,          # [batch_size]
        z_coord,           # [batch_size]
        src_key_padding_mask=None,
        tgt_key_padding_mask=None
    ):
        # Embed source (2D) features including pressure
        src_embedded = self.source_embedding(src_features)
        # Embed target (3D) geometric features
        tgt_embedded = self.target_embedding(tgt_geometry)
        
        # Embed and add global features
        global_embedded = self.global_embedding(mach, reynolds, z_coord)
        global_embedded = global_embedded.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Add global features to both source and target embeddings
        src_embedded = src_embedded + global_embedded
        tgt_embedded = tgt_embedded + global_embedded
        
        # Pass through decoder
        output = self.decoder(
            tgt=tgt_embedded,
            memory=src_embedded,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to output dimension
        return self.output_projection(output)

class AirfoilTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enable_transfer_learning = config.enable_transfer_learning

        self.decoder = AirfoilTransformerDecoder(
            enable_transfer_learning = config.enable_transfer_learning,
            d_model=config.d_model,
            nhead=config.n_head,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout
        )
    
    def forward(
        self,
        src_spline_features,
        tgt_spline_features,
        mach,
        reynolds,
        z_coord,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None
    ):
        if not self.enable_transfer_learning:
            # Modify input processing if needed when not using transfer learning
            # Attend only to geometric features
            src_spline_features = tgt_spline_features  # Only use geometry, not pressure
        return self.decoder(
            src_spline_features,
            tgt_spline_features,
            mach,
            reynolds,
            z_coord,
            #src_key_padding_mask,
            #tgt_key_padding_mask
        )