import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import sys


class AirfoilDataset(Dataset):
    """Dataset for 2D to 3D airfoil pressure mapping"""

    def __init__(self, data_path, split="train"):
        """
        Args:
            data_path (str): Path to the data file
            split (str): One of 'train', 'val', or 'test'
        """
        super().__init__()
        self.data_path = data_path
        self.split = split

        print(self.data_path)
        print(self.split)

        fname = "../" + self.data_path + "/" + self.split + "_aero_2d_data.pkl"
        with open(fname, "rb") as handle:
            tl_data = pickle.load(handle)

            # Calculate total amount of data in split
            n_split_data = 0
            for keys, values in tl_data.items():
                n_split_data += values["2D"][0].shape[0]

            # Get feature dimension information
            ncases = n_split_data
            nsequence = 184
            nsection = 1
            N = nsection * ncases

            # Initialize arrays
            airfoil_2d = np.zeros((N, nsequence, 3))
            mach, reynolds, aoa = np.zeros(N), np.zeros(N), np.zeros(N)
            case_id = np.zeros(N, dtype=np.uint32)

            idx = 0
            for i, (key, value) in enumerate(tl_data.items()):
                # Get data
                data_2d = value["2D"]
                geo_2d, field_2d = data_2d
                nslice = geo_2d.shape[0]

                ma, re, alpha = value["mach"], value["reynolds"], value["alpha"]

                airfoil_2d[idx : idx + nslice, ...] = np.concatenate(
                    (
                        geo_2d[..., :2],
                        np.expand_dims(field_2d[..., 0], -1),
                    ),
                    axis=-1,
                )
                mach[idx : idx + nslice] = ma
                reynolds[idx : idx + nslice] = re
                aoa[idx : idx + nslice] = alpha
                case_id[idx : idx + nslice] = int(key)
                idx += nslice

            self.airfoil_2d = torch.from_numpy(airfoil_2d)
            self.mach = torch.from_numpy(mach)
            self.reynolds = torch.from_numpy(reynolds)
            self.aoa = torch.from_numpy(aoa)
            self.case_id = case_id

    def __len__(self):
        return len(self.airfoil_2d)

    def __getitem__(self, idx):
        return {
            "airfoil_2d": self.airfoil_2d[idx].float(),
            "mach": self.mach[idx].float(),
            "reynolds": self.reynolds[idx].float(),
            "case_id": self.case_id[idx],
        }


def create_dataloaders(data_path, batch_size=32, num_workers=1):
    """
    Create DataLoaders for train, validation, and test sets

    Args:
        data_path (str): Path to the data file
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading

    Returns:
        dict: Dictionary containing train, val, and test DataLoaders
    """
    datasets = {
        split: AirfoilDataset(data_path, split=split)
        for split in ["train", "val", "test"]
    }

    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for split, dataset in datasets.items()
    }

    return dataloaders


class AirfoilDataScaler:
    """
    A class to handle both MinMax and Standard scaling for airfoil data.
    Maintains separate statistics for different feature types.
    """

    def __init__(self):
        self.scalers = {}

    def fit(self, data_loader):
        """
        Compute scaling statistics from the training data
        """
        # Initialize statistics containers
        stats = {
            "pressure_2d": {
                "mean": 0.0,
                "std": 1.0,
                "min": float("inf"),
                "max": float("-inf"),
            },  # MinMax/Std for 2D pressure
            "coord_2d": {
                "mean": 0.0,
                "std": 1.0,
                "min": float("inf"),
                "max": float("-inf"),
            },  # MinMax/Std for 2D coordinates
            "mach": {
                "mean": 0.0,
                "std": 1.0,
                "min": float("inf"),
                "max": float("-inf"),
            },  # MinMax/Std for Mach
            "reynolds": {
                "mean": 0.0,
                "std": 1.0,
                "min": float("inf"),
                "max": float("-inf"),
            },  # MinMax/Std for Reynolds
            "aoa": {
                "mean": 0.0,
                "std": 1.0,
                "min": float("inf"),
                "max": float("-inf"),
            },  # MinMax/Std for Angle of Attack
        }

        n_samples = 0

        # First pass: compute means for standard scaling
        sum_pressure_2d = 0.0
        sum_coords_2d = 0.0
        sum_mach = 0.0
        sum_reynolds = 0.0
        sum_aoa = 0.0

        for batch in data_loader:
            # Get bacth data
            p_2d = batch["airfoil_2d"][..., 2]  # Last channel is pressure
            coords_2d = batch["airfoil_2d"][..., :2]  # x,y coordinates
            mach = batch["mach"]
            reynolds = batch["reynolds"]
            aoa = np.ones_like(mach) * 2.5  # Angle of attack (Fix hard coded aoa)

            # Update data min/max
            stats["pressure_2d"]["min"] = min(
                stats["pressure_2d"]["min"], p_2d.min().item()
            )
            stats["pressure_2d"]["max"] = max(
                stats["pressure_2d"]["max"], p_2d.max().item()
            )

            stats["coord_2d"]["min"] = min(
                stats["coord_2d"]["min"], coords_2d.min().item()
            )
            stats["coord_2d"]["max"] = max(
                stats["coord_2d"]["max"], coords_2d.max().item()
            )

            stats["mach"]["min"] = min(stats["mach"]["min"], mach.min().item())
            stats["mach"]["max"] = max(stats["mach"]["max"], mach.max().item())

            stats["reynolds"]["min"] = min(
                stats["reynolds"]["min"], reynolds.min().item()
            )
            stats["reynolds"]["max"] = max(
                stats["reynolds"]["max"], reynolds.max().item()
            )

            stats["aoa"]["min"] = min(stats["aoa"]["min"], aoa.min().item())
            stats["aoa"]["max"] = max(stats["aoa"]["max"], aoa.max().item())

            # Update data sums
            sum_pressure_2d += p_2d.sum()
            sum_coords_2d += coords_2d.sum()
            sum_mach += mach.sum()
            sum_reynolds += reynolds.sum()
            sum_aoa += aoa.sum()

            n_samples += len(batch["airfoil_2d"])

        # Compute means
        ns = p_2d.shape[-1]
        stats["pressure_2d"]["mean"] = (
            sum_pressure_2d / (n_samples * ns)
        ).item()  # 200 points for pressure
        stats["coord_2d"]["mean"] = (
            sum_coords_2d / (n_samples * ns)
        ).item()  # 200 points for pressure
        stats["mach"]["mean"] = (sum_mach / n_samples).item()
        stats["reynolds"]["mean"] = (sum_reynolds / n_samples).item()
        stats["aoa"]["mean"] = (sum_aoa / n_samples).item()

        # Second pass: compute standard deviations
        sum_sq_pressure_2d = 0.0
        sum_sq_coords_2d = 0.0
        sum_sq_mach = 0.0
        sum_sq_reynolds = 0.0
        sum_sq_aoa = 0.0

        for batch in data_loader:
            # Get bacth data
            p_2d = batch["airfoil_2d"][..., 2]  # Last channel is pressure
            coords_2d = batch["airfoil_2d"][..., :2]  # x,y coordinates
            mach = batch["mach"]
            reynolds = batch["reynolds"]
            aoa = np.ones_like(mach) * 2.5  # Angle of attack (Fix hard coded aoa)

            # Update data variance
            sum_sq_pressure_2d += ((p_2d - stats["pressure_2d"]["mean"]) ** 2).sum()
            sum_sq_coords_2d += ((coords_2d - stats["coord_2d"]["mean"]) ** 2).sum()
            sum_sq_mach += ((mach - stats["mach"]["mean"]) ** 2).sum()
            sum_sq_reynolds += ((reynolds - stats["reynolds"]["mean"]) ** 2).sum()
            sum_sq_aoa += ((aoa - stats["aoa"]["mean"]) ** 2).sum()

        # Compute standard deviations
        stats["pressure_2d"]["std"] = np.sqrt(
            sum_sq_pressure_2d / (n_samples * ns)
        ).item()
        stats["coord_2d"]["std"] = np.sqrt(sum_sq_coords_2d / (n_samples * ns)).item()
        stats["mach"]["std"] = np.sqrt(sum_sq_mach / n_samples).item()
        stats["reynolds"]["std"] = np.sqrt(sum_sq_reynolds / n_samples).item()
        stats["aoa"]["std"] = np.sqrt(sum_sq_aoa / n_samples).item()

        self.scalers = stats
        return self

    def transform(self, batch):
        """
        Scale a batch of data using the computed statistics
        """
        scaled_batch = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        # Scale Reynolds number (Standard)
        scaled_batch["reynolds"] = (
            batch["reynolds"] - self.scalers["reynolds"]["mean"]
        ) / self.scalers["reynolds"]["std"]

        return scaled_batch

    def inverse_transform(self, batch):
        """
        Inverse scale the transformed data
        """
        unscaled_batch = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        # Unscale Reynolds number
        unscaled_batch["reynolds"] = (
            batch["reynolds"] * self.scalers["reynolds"]["std"]
            + self.scalers["reynolds"]["mean"]
        )

        return unscaled_batch

    def save(self, path):
        """Save scaler statistics"""
        torch.save(self.scalers, path)

    def load(self, path):
        """Load scaler statistics"""
        self.scalers = torch.load(path)
        return self
