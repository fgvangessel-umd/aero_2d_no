import torch
import torch.nn as nn
import math


class GlobalFeatureEmbedding(nn.Module):
    """Embeds global features (Mach, Reynolds) into the model dimension"""

    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Linear(2, d_model)

    def forward(self, mach, reynolds):
        global_features = torch.stack([mach, reynolds], dim=-1)
        return self.embedding(global_features)


class SplineFeatureEmbedding(nn.Module):
    """Projects spline features into the model dimension"""

    def __init__(self, d_model, include_pressure=True):
        super().__init__()
        self.input_dim = 3  # 3 features: arc_length, x, y
        self.embedding = nn.Linear(self.input_dim, d_model)

    def forward(self, spline_features):
        return self.embedding(spline_features)


class AirfoilTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        # Embeddings for 2D geometry
        self.geo_embedding = SplineFeatureEmbedding(d_model, include_pressure=False)
        self.global_embedding = GlobalFeatureEmbedding(d_model)

        # Create decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Stack decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection to single pressure value
        self.output_projection = nn.Linear(d_model, 1)  # Project to pressure only

    def forward(
        self,
        tgt_spline_features,  # [batch_size, seq_len, 3] - 2D airfoil
        mem_spline_features,  # [batch_size, seq_len, 3] - 2D airfoil
        mach,  # [batch_size]
        reynolds,  # [batch_size]
    ):
        # Embed source (2D) features including pressure
        geo_tgt_embedded = self.source_embedding(tgt_spline_features)
        geo_mem_embedded = self.source_embedding(mem_spline_features)

        # Embed and add global features
        global_embedded = self.global_embedding(mach, reynolds)
        global_embedded = global_embedded.unsqueeze(1)  # [batch_size, 1, d_model]

        # Add global features to both memory and target embeddings
        seq_tgt_embedded = geo_tgt_embedded + global_embedded
        seq_mem_embedded = geo_mem_embedded + global_embedded

        # Pass through decoder
        output = self.decoder(
            tgt=seq_tgt_embedded,
            memory=seq_mem_embedded,
        )

        # Project to output dimension
        return self.output_projection(output)


class AirfoilTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = AirfoilTransformerDecoder(
            d_model=config.d_model,
            nhead=config.n_head,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
        )

    def forward(
        self,
        geo_features,
        mach,
        reynolds,
    ):
        # Use same sequence for target and memory (i.e. self-attention)
        tgt_spline_features = geo_features  # Only use geometry
        mem_spline_features = geo_features  # Only use geometry

        return self.decoder(
            tgt_spline_features,
            mem_spline_features,
            mach,
            reynolds,
        )
