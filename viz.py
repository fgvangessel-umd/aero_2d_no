import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

def generate_colors_from_zcoords(z_coords, cmap_name='viridis', alpha=1.0):
    """
    Generates colors based on z-coordinate values using a specified colormap.
    
    Args:
        z_coords: Array of z-coordinate values
        cmap_name: Name of the colormap to use (default: 'viridis')
        alpha: Transparency value between 0 and 1 (default: 1.0)
        
    Returns:
        colors_array: Array of RGBA colors corresponding to z_coords
        norm: Normalization object used for colorbar creation
        cmap: Colormap object for reference
    """
    # Normalize z-coordinates
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    norm = colors.Normalize(vmin=z_min, vmax=z_max)
    
    # Create colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Map z-coordinates to colors
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    colors_array = mapper.to_rgba(z_coords, alpha=alpha)
    
    return colors_array, norm, cmap

# Example for creating a colorbar separately:
def add_zcoord_colorbar(fig, ax, norm, cmap, label='Z-Coordinate'):
    """Adds a colorbar based on z-coordinate normalization"""
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(label)
    return cbar

def visualize_predictions(model, dataloader, scaler, device, num_samples=1, save_path='predictions'):
    """
    Visualize model predictions against ground truth
    
    Args:
        model: Trained AirfoilTransformerModel
        dataloader: DataLoader containing test data
        scaler: Fitted AirfoilDataScaler
        device: torch device
        num_samples: Number of random samples to visualize
        save_path: Directory to save visualization plots
    """
    model.eval()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        # Move data to device
        airfoil_2d = batch['airfoil_2d'].to(device)
        geometry_3d = batch['geometry_3d'].to(device)
        pressure_3d_true = batch['pressure_3d'].to(device)
        mach = batch['mach'].to(device)
        reynolds = batch['reynolds'].to(device)
        z_coord = batch['z_coord'].to(device)
        case_ids = batch['case_id']
        
        # Scale reynolds number as in training
        reynolds = (reynolds - scaler['reynolds']['mean']) / scaler['reynolds']['std']
        
        # Get model predictions
        pressure_3d_pred = model(
            airfoil_2d,
            geometry_3d,
            mach,
            reynolds,
            z_coord
        )
        
        # Store ground-truth and prediction data
        batch_pred = {
            'pressure_3d': pressure_3d_pred,
            'airfoil_2d': airfoil_2d,
            'geometry_3d': geometry_3d,
            'mach': mach,
            'reynolds': reynolds
        }
        batch_true = {
            'pressure_3d': pressure_3d_true,
            'airfoil_2d': airfoil_2d,
            'geometry_3d': geometry_3d,
            'mach': mach,
            'reynolds': reynolds
        }
        
        # Visualize random samples
        for i in range(min(num_samples, len(case_ids))):
            fig = plt.figure(figsize=(20, 15))
            gs = GridSpec(2, 2, figure=fig)
            
            # 1. Geometric Configuration (2D & 3D)
            ax_geo = fig.add_subplot(gs[0, 0])
            x_2d = airfoil_2d[i,:,1].cpu().numpy()
            y_2d = airfoil_2d[i,:,2].cpu().numpy()
            x_3d = geometry_3d[i,:,1].cpu().numpy()
            y_3d = geometry_3d[i,:,2].cpu().numpy()
            
            ax_geo.scatter(x_2d, y_2d, c='blue', label='2D Profile', alpha=0.6)
            ax_geo.scatter(x_3d, y_3d, c='red', label='3D Section', alpha=0.6)
            ax_geo.set_title(f'Geometric Configuration\nz = {z_coord[i].item():.3f}', fontsize=20)
            ax_geo.set_aspect('equal')
            ax_geo.legend(fontsize=18)
            ax_geo.grid(True)
            
            # 2. Pressure Distribution
            ax_press = fig.add_subplot(gs[0, 1])
            p_2d = batch_true['airfoil_2d'][i,:,3].cpu().numpy()
            p_3d_true = batch_true['pressure_3d'][i,:,0].cpu().numpy()
            p_3d_pred = batch_pred['pressure_3d'][i,:,0].cpu().numpy()
            
            ax_press.plot(x_2d, p_2d, c='k', linestyle='-.', label='2D Pressure', alpha=0.6)
            ax_press.scatter(x_3d, p_3d_true, c='black', label='3D True', alpha=0.6)
            ax_press.scatter(x_3d, p_3d_pred, c='red', label='3D Predicted', alpha=0.6)
            ax_press.set_title('Pressure Distribution', fontsize=20)
            ax_press.legend(fontsize=18)
            ax_press.grid(True)
            
            # 3. Pressure Difference (True vs Predicted)
            ax_diff = fig.add_subplot(gs[1, 0])
            pressure_diff = p_3d_pred - p_3d_true
            ax_diff.scatter(x_3d, pressure_diff, c='purple', alpha=0.6)
            ax_diff.set_title('Prediction Error', fontsize=20)
            ax_diff.axhline(y=0, color='k', linestyle='--')
            ax_diff.grid(True)
            
            # 4. Error Distribution
            ax_hist = fig.add_subplot(gs[1, 1])
            ax_hist.hist(pressure_diff, bins=30, alpha=0.6, color='purple')
            ax_hist.set_title('Error Distribution', fontsize=20)
            ax_hist.axvline(x=0, color='k', linestyle='--')
            ax_hist.grid(True)
            
            # Add case information
            fig.suptitle(f'Case ID: {case_ids[i].item()}\n' + 
                        f'Mach: {mach[i].item():.3f}, ' +
                        f'Reynolds: {batch_true["reynolds"][i].item():.2e}',
                        fontsize=24)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/case_{case_ids[i].item()}_z_{z_coord[i].item():.3f}.png')
            plt.close()

def plot_3d_wing_predictions(xy_2d, xy_3d, p_2d, p_3d_true, p_3d_pred, z_coord, case_data, fname):
    """
    Visualize 3D wing pressure predictions against ground truth
    
    Args:
        xy_2d: array of 2D airfoil coordinates [Nsection, Nspline, 2]
        xy_3d: array of 3D cross-section coordinates [Nsection, Nspline, 2]
        p_2d: array of 2D airfoil pressures [Nsection, Nspline]
        p_3d_true: array of ground truth 3D cross-section pressures [Nsection, Nspline]
        p_3d_preds: array of predicted 3D cross-section pressures [Nsection, Nspline]
        z_coord: array of z-coordinate of the cross-section [Nsection]
        case_data: Dictionary of case ID, Mach number, and Reynolds number
        fname: string of figure save location
    """
    # Visualize
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, figure=fig)

    # Create colorbar
    colors_array, norm, cmap = generate_colors_from_zcoords(z_coord, cmap_name='RdBu_r', alpha=1.0)
    
    # 1. Geometric Configuration (2D & 3D)
    ax_geo = fig.add_subplot(gs[0])
    
    ax_geo.plot(xy_2d[0,:, 0], xy_2d[0,:,1], c='k', alpha=0.6)
    ax_geo.set_title(f'Wing Cross-Section Geometry', fontsize=22)
    ax_geo.set_aspect('equal')
    ax_geo.grid(True)
    
    # 2. Pressure Distribution
    ax_press = fig.add_subplot(gs[1])

    ax_press.plot(xy_2d[0,:, 0], p_2d[0,:], c='k', linestyle=':', label='2D Pressure', alpha=0.6)

    for i in range(p_3d_pred.shape[0]):
        if i==0:
            ax_press.plot(xy_3d[i,:,0], p_3d_true[i,:], c=colors_array[i], label='3D True', alpha=0.8)
            ax_press.plot(xy_3d[i,:,0], p_3d_pred[i,:], c=colors_array[i], linestyle='-.', label='3D Predicted', alpha=0.8)
        else:
            ax_press.plot(xy_3d[i,:,0], p_3d_true[i,:], c=colors_array[i], alpha=0.8)
            ax_press.plot(xy_3d[i,:,0], p_3d_pred[i,:], c=colors_array[i], linestyle='-.', alpha=0.8)

    # Create ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax_press)
    cbar.set_label('Z-Coordinate', fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    ax_press.set_title('Pressure Distribution', fontsize=22)
    ax_press.legend(fontsize=22)
    ax_press.grid(True)
    
    # Add case information
    fig.suptitle(f"Case ID: {case_data['case_id']}\n" + 
                f"Mach: {case_data['mach']:.3f}, " +
                f"Reynolds: {case_data['reynolds']:.3e}",
                fontsize=24)
    
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    return None