import yaml
from dataclasses import dataclass, asdict
import argparse
from typing import List, Optional
from datetime import datetime
import os


@dataclass
class TrainingConfig:
    # Model architecture
    d_model: int = 256
    n_head: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    # Transfer learning parameters
    enable_transfer_learning: bool = (
        True  # Default to True to maintain current behavior
    )

    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 1
    weight_decay: float = 0.0
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    checkpoint_freq: int = 10
    num_output_figs: int = 4
    validation_freq: int = 1  # Validate every epoch
    test_freq: int = 10  # Test every 5 epochs
    viz_freq: int = 10  # Visualize every 5 epochs

    # Hardware parameters
    device: Optional[str] = None  # 'cuda', 'mps', 'cpu', or None for auto-detection

    # Data parameters
    data_path: str = "data"
    num_workers: int = 1
    scaler_fname: str = ""

    # Experiment tracking
    project_name: str = "2d-airfoil-no"
    experiment_name: str = "baseline"
    tags: Optional[List[str]] = None

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_args(cls, args=None):
        """
        Create config from command-line arguments with optional YAML config file.
        Command-line arguments override YAML settings if both are provided.
        """
        parser = argparse.ArgumentParser(description="Training configuration")

        # Allow loading base config from YAML
        parser.add_argument("--config", type=str, help="Path to YAML config file")

        # Model architecture params
        parser.add_argument("--d_model", type=int, help="Model dimension")
        parser.add_argument("--n_head", type=int, help="Number of attention heads")
        parser.add_argument("--n_layers", type=int, help="Number of transformer layers")
        parser.add_argument("--d_ff", type=int, help="Feed-forward network dimension")
        parser.add_argument("--dropout", type=float, help="Dropout rate")

        # Training params
        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument("--learning_rate", type=float, help="Learning rate")
        parser.add_argument("--num_epochs", type=int, help="Number of epochs")
        parser.add_argument("--weight_decay", type=float, help="Weight decay")
        parser.add_argument(
            "--scheduler_type", type=str, help="Learning rate scheduler type"
        )
        parser.add_argument("--warmup_steps", type=int, help="Warmup steps")
        parser.add_argument("--checkpoint_freq", type=int, help="Checkpoint frequency")

        # Hardware params
        parser.add_argument(
            "--device",
            type=str,
            choices=["cuda", "mps", "cpu"],
            help="Device to use (cuda, mps for Apple Silicon, or cpu)",
        )

        # Data params
        parser.add_argument("--data_path", type=str, help="Path to data files")
        parser.add_argument(
            "--num_workers", type=int, help="Number of data loader workers"
        )
        parser.add_argument("--scaler_fname", type=str, help="Scaler filename")

        # Experiment tracking
        parser.add_argument("--project_name", type=str, help="W&B project name")
        parser.add_argument("--experiment_name", type=str, help="Experiment name")
        parser.add_argument(
            "--tags", type=str, nargs="+", help="Tags for the experiment"
        )
        parser.add_argument("--notes", type=str, help="Notes for the experiment")

        # Parse args
        parsed_args = parser.parse_args(args)

        # Initialize config
        config = cls()

        # Load from YAML if provided
        if parsed_args.config:
            config = cls.load(parsed_args.config)

        # Override with command-line args (if provided)
        arg_dict = vars(parsed_args)
        for key, value in arg_dict.items():
            if key != "config" and value is not None:
                if hasattr(config, key):
                    setattr(config, key, value)

        return config
