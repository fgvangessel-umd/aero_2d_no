import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from utils import calculate_airfoil_forces
import sys


@dataclass
class ValidationMetrics:
    """Container for validation metrics"""

    mse: float  # Mean squared error
    rmse: float  # Root mean squared error
    mae: float  # Mean absolute error
    max_error: float  # Maximum absolute error
    r2_score: float  # R-squared score
    pressure_correlation: float  # Correlation coefficient for pressure


class ModelValidator:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: Optional[torch.device] = None,
        scaler: Optional[object] = None,
    ):
        """
        Initialize the model validator

        Args:
            model: The transformer model to validate
            criterion: Loss function
            device: Computation device (CPU/GPU/MPS). If None, will be auto-selected.
            scaler: Data scaler for normalization
        """
        # Set device if not provided - support CUDA, MPS (Apple Silicon), or CPU
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.model = model
        self.criterion = criterion
        self.device = device
        self.scaler = scaler

    def analyze_spanwise_distribution(
        self,
        pred_pressure: torch.Tensor,
        true_pressure: torch.Tensor,
        z_coordinates: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze error distribution along the wing span

        Args:
            pred_pressure: Predicted pressure values
            true_pressure: Ground truth pressure values
            z_coordinates: Spanwise coordinates

        Returns:
            Dictionary containing spanwise error distributions
        """
        errors = (pred_pressure - true_pressure).abs()

        # Sort by z-coordinate
        sorted_idx = torch.argsort(z_coordinates)
        sorted_errors = errors[sorted_idx]
        sorted_z = z_coordinates[sorted_idx]

        return {
            "z_coordinates": sorted_z.cpu().numpy(),
            "error_distribution": sorted_errors.cpu().numpy(),
        }

    def validate_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Validate a single batch of data

        Args:
            batch: Dictionary containing batch data

        Returns:
            Tuple of (predictions, batch metrics)
        """
        self.model.eval()

        with torch.no_grad():
            # Scale batch if scaler is provided
            if self.scaler:
                batch = self.scaler.transform(batch)

            # Move data to device
            airfoil_2d = batch["airfoil_2d"].to(self.device)
            mach = batch["mach"].to(self.device)
            reynolds = batch["reynolds"].to(self.device)
            geo_2d = airfoil_2d[:, :, :-1]
            pressure_2d = airfoil_2d[:, :, -1].unsqueeze(-1)

            # Forward pass
            predictions = self.model(geo_2d, mach, reynolds)

            # Compute basic metrics
            loss = self.criterion(predictions, pressure_2d)
            mae = torch.nn.functional.l1_loss(predictions, pressure_2d)
            max_error = (predictions - pressure_2d).abs().max()

            batch_metrics = {
                "batch_loss": loss.item(),
                "batch_mae": mae.item(),
                "batch_max_error": max_error.item(),
            }

            # print(batch_metrics)

            return predictions, batch_metrics

    def validate_dataset(
        self, dataloader: DataLoader, validation_type: str = "val"
    ) -> ValidationMetrics:
        """
        Validate the entire dataset

        Args:
            dataloader: DataLoader containing validation data
            epoch: Current epoch number for logging

        Returns:
            ValidationMetrics object containing computed metrics
        """
        all_predictions = []
        all_true_values = []
        all_z_coords = []
        total_metrics = {}

        for batch in dataloader:
            predictions, batch_metrics = self.validate_batch(batch)

            airfoil_2d = batch["airfoil_2d"].to(self.device)
            pressure_2d = airfoil_2d[:, :, -1].unsqueeze(-1)

            # Collect predictions and true values
            all_predictions.append(predictions)
            all_true_values.append(pressure_2d)

            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)

        # Concatenate all predictions and true values
        all_predictions = torch.cat(all_predictions)
        all_true_values = torch.cat(all_true_values)

        # Compute aggregate metrics
        mse = torch.nn.functional.mse_loss(all_predictions, all_true_values).item()
        rmse = np.sqrt(mse)
        mae = torch.nn.functional.l1_loss(all_predictions, all_true_values).item()
        max_error = (all_predictions - all_true_values).abs().max().item()

        # Compute R² score
        r2 = r2_score(
            all_true_values.cpu().numpy().flatten(),
            all_predictions.cpu().numpy().flatten(),
        )

        # Compute pressure correlation
        correlation = np.corrcoef(
            all_true_values.cpu().numpy().flatten(),
            all_predictions.cpu().numpy().flatten(),
        )[0, 1]

        # Create ValidationMetrics object
        metrics = ValidationMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            r2_score=r2,
            pressure_correlation=correlation,
        )

        # Log to W&B if enabled
        # if self.log_to_wandb and global_step is not None:
        #    self._log_to_wandb(metrics, global_step)

        return metrics

    '''
    def _log_to_wandb(self, metrics: ValidationMetrics, global_step: int):
        """Log validation metrics to W&B"""
        log_dict = {
            "val/mse": metrics.mse,
            "val/rmse": metrics.rmse,
            "val/mae": metrics.mae,
            "val/max_error": metrics.max_error,
            "val/r2_score": metrics.r2_score,
            "val/pressure_correlation": metrics.pressure_correlation,
            "step": global_step,
        }

        # Log section-wise metrics
        for section, value in metrics.section_wise_errors.items():
            log_dict[f"val/section/{section}"] = value

        # Create and log spanwise distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            np.repeat(
                metrics.spanwise_distribution["z_coordinates"][:, np.newaxis],
                metrics.spanwise_distribution["error_distribution"].shape[1],
                axis=1,
            ),
            metrics.spanwise_distribution["error_distribution"].squeeze(),
            alpha=0.5,
        )
        ax.set_xlabel("Spanwise Position (z)")
        ax.set_ylabel("Absolute Error")
        ax.set_title("Error Distribution Along Wing Span")

        log_dict["val/spanwise_distribution"] = wandb.Image(fig)
        plt.close(fig)

        wandb.log(log_dict)'
    '''

    def generate_validation_report(
        self, metrics: ValidationMetrics, save_path: Optional[str] = None
    ) -> str:
        """
        Generate a detailed validation report

        Args:
            metrics: ValidationMetrics object
            save_path: Optional path to save the report

        Returns:
            Formatted report string
        """
        report = [
            "Validation Report",
            "================",
            f"Mean Squared Error (MSE): {metrics.mse:.6f}",
            f"Root Mean Squared Error (RMSE): {metrics.rmse:.6f}",
            f"Mean Absolute Error (MAE): {metrics.mae:.6f}",
            f"Maximum Error: {metrics.max_error:.6f}",
            f"R² Score: {metrics.r2_score:.6f}",
            f"Pressure Correlation: {metrics.pressure_correlation:.6f}",
            "",
            "Section-wise Metrics:",
            "-------------------",
        ]

        for section, error in metrics.section_wise_errors.items():
            report.append(f"{section}: {error:.6f}")

        report = "\n".join(report)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)

        return report

    def compute_force_metrics(
        self,
        pred_pressure: torch.Tensor,
        true_pressure: torch.Tensor,
        geometry: torch.Tensor,
        alpha: float = 2.5,
    ) -> Dict[str, float]:
        """
        Compute lift and drag metrics from pressure predictions

        Returns:
            Dictionary containing lift/drag metrics
        """
        # Convert tensors to numpy arrays
        pred_p = pred_pressure.cpu().numpy()
        true_p = true_pressure.cpu().numpy()
        xy = geometry.cpu().numpy()
        nsections = xy.shape[0]

        # Calculate forces for each airfoil section
        pred_section_lifts, pred_section_drags = (
            np.zeros(nsections),
            np.zeros(nsections),
        )
        true_section_lifts, true_section_drags = (
            np.zeros(nsections),
            np.zeros(nsections),
        )

        for i in range(xy.shape[0]):
            _, pred_force = calculate_airfoil_forces(
                xy[i], pred_p[i].squeeze(), alpha=np.deg2rad(alpha)
            )
            _, true_force = calculate_airfoil_forces(
                xy[i], true_p[i].squeeze(), alpha=np.deg2rad(alpha)
            )

            pred_section_lifts[i] = pred_force[1]  # Lift is y-component
            pred_section_drags[i] = pred_force[0]  # Drag is x-component
            true_section_lifts[i] = true_force[1]
            true_section_drags[i] = true_force[0]

        # Calculate total lifts and drags for entire wing
        pred_wing_lift, pred_wing_drag = (
            np.sum(pred_section_lifts),
            np.sum(pred_section_drags),
        )
        true_wing_lift, true_wing_drag = (
            np.sum(true_section_lifts),
            np.sum(true_section_drags),
        )

        return {
            "true_section_lifts": true_section_lifts,
            "pred_section_lifts": pred_section_lifts,
            "true_section_drags": true_section_drags,
            "pred_section_drags": pred_section_drags,
            "true_wing_lift": true_wing_lift,
            "pred_wing_lift": pred_wing_lift,
            "true_wing_drag": true_wing_drag,
            "pred_wing_drag": pred_wing_drag,
        }
