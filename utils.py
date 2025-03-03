import torch
import numpy as np
from pathlib import Path
from typing import Tuple

def calculate_airfoil_forces(
    points: np.ndarray,
    pressures: np.ndarray,
    alpha: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate nodal and total forces on an airfoil given discrete points and pressure values.
    
    Parameters
    ----------
    points : np.ndarray
        Nx2 array of (x,y) coordinates along the airfoil surface
    pressures : np.ndarray
        Length N array of pressure values at each point
    alpha : float
        Angle of attack of airfoil
        
    Returns
    -------
    nodal_forces : np.ndarray
        Nx2 array of force vectors at each node
    total_force : np.ndarray
        2-element array containing the total force vector [Fx, Fy]
        
    Notes
    -----
    Forces are calculated using a piecewise linear approximation between points.
    The direction of the force is determined by the local normal vector.
    """
    
    # Input validation
    if points.shape[0] != pressures.shape[0]:
        raise ValueError("Number of points must match number of pressure values")
    if points.shape[1] != 2:
        raise ValueError("Points must be 2D coordinates")
        
    N = points.shape[0]
    
    # Calculate vectors between adjacent points
    # Use periodic boundary for last point
    delta_points = np.roll(points, -1, axis=0) - points
    
    # Calculate length of each segment
    segment_lengths = np.sqrt(np.sum(delta_points**2, axis=1))

    # Calculate normal vectors (rotate tangent vector 90 degrees counterclockwise)
    normal_vectors = np.zeros_like(points)
    normal_vectors[:, 0] = -delta_points[:, 1] / segment_lengths  # Fixed broadcasting
    normal_vectors[:, 1] = delta_points[:, 0] / segment_lengths   # Fixed broadcasting
    
    # Calculate average pressure for each segment
    # Average between current and next point
    segment_pressures = 0.5 * (pressures + np.roll(pressures, -1))
    
    # Calculate segment forces
    # Force = pressure * length * normal_vector
    segment_forces = normal_vectors * segment_pressures[:, np.newaxis] * segment_lengths[:, np.newaxis]

    # Define rotation array to transform from airfoil affixed coordinate system into free-stream coordinate system
    R = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    segment_forces = segment_forces @ R.T
    
    # Calculate total force by summing all segment forces
    total_force = np.sum(segment_forces, axis=0)
    
    return segment_forces, total_force

def select_batches(tensor, batch_indices):
    """
    Select specific batches from a 3D tensor
    
    Args:
        tensor: A 3D PyTorch tensor of shape [batch_size, dim1, dim2]
        batch_indices: List or tensor of indices to select from batch dimension
        
    Returns:
        A tensor containing only the selected batches
    """
    # Convert indices to tensor if they're in a list
    if isinstance(batch_indices, list):
        batch_indices = torch.tensor(batch_indices, device=tensor.device)
        
    # Select the specified batches
    selected_batches = tensor[batch_indices]
    
    return selected_batches

def select_case(batch, true_reynolds, case_id, device):
    """
    Takes a min-batch of aero data and returns subset of data consistenting to a single case 
    (unique combination of geometry, mach number, and reynolds number)

    batch: dictionary of numpy arrays. The first index of these arrays is the batch index, indexing each cross section
    case_id: integer ID of case to isolate and return
    """
    # Move data to device
    airfoil_2d = batch['airfoil_2d'].to(device)
    geometry_3d = batch['geometry_3d'].to(device)
    pressure_3d = batch['pressure_3d'].to(device)
    mach = batch['mach'].to(device)
    reynolds = batch['reynolds'].to(device)
    z_coord = batch['z_coord'].to(device)
    cases = batch['case_id'].numpy().tolist()

    # Select data corresponding to a single wing
    idxs = [i for i, x in enumerate(cases) if x == case_id]
    try:
        airfoil_2d = select_batches(airfoil_2d, idxs)
        geometry_3d = select_batches(geometry_3d, idxs)
        pressure_3d = select_batches(pressure_3d, idxs)
        mach = select_batches(mach, idxs)
        reynolds = select_batches(reynolds, idxs)
        true_reynolds = select_batches(true_reynolds, idxs)
        z_coord = select_batches(z_coord, idxs)
    except IndexError:
        sys.exit("Requested Case does not exist in batch")

    return airfoil_2d, geometry_3d, pressure_3d, mach, reynolds, true_reynolds, z_coord


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """
    Load model checkpoint and restore scaler state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: PyTorch optimizer
        scaler: AirfoilDataScaler instance
    """
    # Use weights_only=False to handle numpy scalars in the checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Update model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler directly if it was saved as a dictionary
    if isinstance(checkpoint['scaler_state'], dict):
        scaler.scalers = checkpoint['scaler_state']
    else:
        # Create temporary file for scaler state
        temp_scaler_path = Path("temp_scaler_load.pt")
        
        # Write scaler state to temporary file
        with open(temp_scaler_path, 'wb') as f:
            f.write(checkpoint['scaler_state'])
        
        # Load scaler state
        scaler.load(temp_scaler_path)
        
        # Clean up temporary file
        if temp_scaler_path.exists():
            temp_scaler_path.unlink()
    
    return model, optimizer, scaler, checkpoint['epoch'], checkpoint['metrics']