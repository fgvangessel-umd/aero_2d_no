import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

class AirfoilDataset(Dataset):
    """Dataset for 2D to 3D airfoil pressure mapping"""
    
    def __init__(self, data_path, split='train'):
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

        fname = self.data_path+'/'+self.split+'_tl_data.pkl'
        with open(fname, 'rb') as handle:
            tl_data = pickle.load(handle)

            # Get feature dimension information
            ncases = len(tl_data.keys())
            nsequence = 200
            nsection = 9
            N = nsection*ncases

            # Initialize arrays
            airfoil_2d  = np.zeros((N, nsequence, 4))
            geometry_3d = np.zeros((N, nsequence, 3))
            pressure_3d = np.zeros((N, nsequence, 1))
            mach, reynolds, z_coord = np.zeros(N), np.zeros(N), np.zeros(N)
            case_id = np.zeros(N, dtype=np.uint32)

            for i, (key, value) in enumerate(tl_data.items()):
                # Get data
                data_2d = value['2D']
                data_3d = value['3D']
                geo_2d, field_2d = data_2d
                geo_3d, field_3d = data_3d

                ma, re = value['mach'], value['reynolds']

                # Fill data arrays
                for j in range(nsection):
                    airfoil_2d[i*nsection+j, ...] = np.concatenate((np.linspace(0.0, 1.0, 200).reshape((-1,1)), \
                                                                    geo_2d[0, :,:2], \
                                                                    field_2d[0, :,0].reshape((-1,1))), axis=1 )
                    geometry_3d[i*nsection+j, ...] =  np.concatenate((np.linspace(0.0, 1.0, 200).reshape((-1,1)), geo_3d[j, :, :2]), axis=1)
                    pressure_3d[i*nsection+j, :, 0] = field_3d[j,:,0]
                    mach[i*nsection+j]     = ma
                    reynolds[i*nsection+j] = re
                    z_coord[i*nsection+j] = geo_3d[j, 0, -1]
                    case_id[i*nsection+j] = int(key)

            self.airfoil_2d = torch.from_numpy(airfoil_2d)
            self.geometry_3d = torch.from_numpy(geometry_3d)
            self.pressure_3d = torch.from_numpy(pressure_3d)
            self.mach = torch.from_numpy(mach)
            self.reynolds = torch.from_numpy(reynolds)
            self.z_coord = torch.from_numpy(z_coord)
            self.case_id = case_id

    def __len__(self):
        return len(self.airfoil_2d)

    def __getitem__(self, idx):
        return {
            'airfoil_2d': self.airfoil_2d[idx].float(),
            'geometry_3d': self.geometry_3d[idx].float(),
            'pressure_3d': self.pressure_3d[idx].float(),
            'mach': self.mach[idx].float(),
            'reynolds': self.reynolds[idx].float(),
            'z_coord': self.z_coord[idx].float(),
            'case_id': self.case_id[idx]
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
        for split in ['train', 'val', 'test']
    }
    
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
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
            'pressure_2d': {'mean': 0.0, 'std': 1.0, 'min': float('inf'), 'max': float('-inf')},  # MinMax/Std for 2D pressure
            'pressure_3d': {'mean': 0.0, 'std': 1.0, 'min': float('inf'), 'max': float('-inf')},  # MinMax/Std for 3D pressure
            'coord_2d': {'mean': 0.0, 'std': 1.0, 'min': float('inf'), 'max': float('-inf')},  # MinMax/Std for 2D coordinates
            'coord_3d': {'mean': 0.0, 'std': 1.0, 'min': float('inf'), 'max': float('-inf')},  # MinMax/Std for 3D coordinates
            'z_coord': {'mean': 0.0, 'std': 1.0, 'min': float('inf'), 'max': float('-inf')},  # MinMax/Std for z-coordinates
            'mach': {'mean': 0.0, 'std': 1.0, 'min': float('inf'), 'max': float('-inf')},  # MinMax/Std for Mach
            'reynolds': {'mean': 0.0, 'std': 1.0, 'min': float('inf'), 'max': float('-inf')},  # MinMax/Std for Reynolds
        }
        
        n_samples = 0
        
        # First pass: compute means for standard scaling
        sum_pressure_2d = 0.0
        sum_pressure_3d = 0.0
        sum_coords_2d = 0.0
        sum_coords_3d = 0.0
        sum_z_coord = 0.0
        sum_mach = 0.0
        sum_reynolds = 0.0
        
        for batch in data_loader:
            # Get bacth data
            p_2d = batch['airfoil_2d'][..., 3]  # Last channel is pressure
            p_3d = batch['pressure_3d']
            coords_2d = batch['airfoil_2d'][..., 1:3]  # x,y coordinates
            coords_3d = batch['geometry_3d'][..., 1:3]  # x,y coordinates
            mach = batch['mach']
            reynolds = batch['reynolds']
            z_coord = batch['z_coord']

            # Update data min/max
            stats['pressure_2d']['min'] = min(stats['pressure_2d']['min'], p_2d.min().item())
            stats['pressure_2d']['max'] = max(stats['pressure_2d']['max'], p_2d.max().item())

            stats['pressure_3d']['min'] = min(stats['pressure_3d']['min'], p_3d.min().item())
            stats['pressure_3d']['max'] = max(stats['pressure_3d']['max'], p_3d.max().item())

            stats['coord_2d']['min'] = min(stats['coord_2d']['min'], coords_2d.min().item())
            stats['coord_2d']['max'] = max(stats['coord_2d']['max'], coords_2d.max().item())

            stats['coord_3d']['min'] = min(stats['coord_3d']['min'], coords_3d.min().item())
            stats['coord_3d']['max'] = max(stats['coord_3d']['max'], coords_3d.max().item())

            stats['z_coord']['min'] = min(stats['z_coord']['min'], z_coord.min().item())
            stats['z_coord']['max'] = max(stats['z_coord']['max'], z_coord.max().item())

            stats['mach']['min'] = min(stats['mach']['min'], mach.min().item())
            stats['mach']['max'] = max(stats['mach']['max'], mach.max().item())

            stats['reynolds']['min'] = min(stats['reynolds']['min'], reynolds.min().item())
            stats['reynolds']['max'] = max(stats['reynolds']['max'], reynolds.max().item())
            
            # Update data sums
            sum_pressure_2d  += p_2d.sum()
            sum_pressure_3d  += p_3d.sum()
            sum_coords_2d += coords_2d.sum() 
            sum_coords_3d += coords_3d.sum()
            sum_z_coord   += z_coord.sum()
            sum_mach      += mach.sum()
            sum_reynolds  += reynolds.sum()
            
            n_samples += len(batch['airfoil_2d'])
        
        # Compute means
        stats['pressure_2d']['mean'] = (sum_pressure_2d / (n_samples * 200)).item()  # 200 points for pressure
        stats['pressure_3d']['mean'] = (sum_pressure_3d / (n_samples * 200)).item()  # 200 points for pressure
        stats['coord_2d']['mean']    = (sum_coords_2d / (n_samples * 200)).item()  # 200 points for pressure
        stats['coord_3d']['mean']    = (sum_coords_3d / (n_samples * 200)).item()  # 200 points for pressure
        stats['z_coord']['mean']     = (sum_z_coord / n_samples).item()
        stats['mach']['mean']        = (sum_mach / n_samples).item()
        stats['reynolds']['mean']    = (sum_reynolds / n_samples).item()
        
        # Second pass: compute standard deviations
        sum_sq_pressure_2d  = 0.0
        sum_sq_pressure_3d  = 0.0
        sum_sq_coords_2d = 0.0
        sum_sq_coords_3d = 0.0
        sum_sq_z_coord    = 0.0
        sum_sq_mach      = 0.0
        sum_sq_reynolds  = 0.0
        
        for batch in data_loader:
            # Get bacth data
            p_2d = batch['airfoil_2d'][..., 3]  # Last channel is pressure
            p_3d = batch['pressure_3d']
            coords_2d = batch['airfoil_2d'][..., 1:3]  # x,y coordinates
            coords_3d = batch['geometry_3d'][..., 1:3]  # x,y coordinates
            mach = batch['mach']
            reynolds = batch['reynolds']
            z_coord = batch['z_coord']

            # Update data variance
            sum_sq_pressure_2d  += ((p_2d - stats['pressure_2d']['mean'])**2).sum()
            sum_sq_pressure_3d  += ((p_3d - stats['pressure_3d']['mean'])**2).sum()
            sum_sq_coords_2d    += ((coords_2d - stats['coord_2d']['mean'])**2).sum()
            sum_sq_coords_3d    += ((coords_3d - stats['coord_3d']['mean'])**2).sum()
            sum_sq_z_coord      += ((z_coord - stats['z_coord']['mean'])**2).sum()
            sum_sq_mach         += ((mach - stats['mach']['mean'])**2).sum()
            sum_sq_reynolds     += ((reynolds - stats['reynolds']['mean'])**2).sum()
        
        # Compute standard deviations
        stats['pressure_2d']['std'] = np.sqrt(sum_sq_pressure_2d / (n_samples * 200)).item()
        stats['pressure_3d']['std'] = np.sqrt(sum_sq_pressure_3d / (n_samples * 200)).item()
        stats['coord_2d']['std'] = np.sqrt(sum_sq_coords_2d / (n_samples * 200)).item()
        stats['coord_3d']['std'] = np.sqrt(sum_sq_coords_3d / (n_samples * 200)).item()
        stats['z_coord']['std'] = np.sqrt(sum_sq_z_coord / n_samples).item()
        stats['mach']['std'] = np.sqrt(sum_sq_mach / n_samples).item()
        stats['reynolds']['std'] = np.sqrt(sum_sq_reynolds / n_samples).item()
        
        self.scalers = stats
        return self
    
    def transform(self, batch):
        """
        Scale a batch of data using the computed statistics
        """
        scaled_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                       for k, v in batch.items()}
        
        # Scale Reynolds number (Standard)
        scaled_batch['reynolds'] = (batch['reynolds'] - self.scalers['reynolds']['mean']) / self.scalers['reynolds']['std']
        
        return scaled_batch
    
    def inverse_transform(self, batch):
        """
        Inverse scale the transformed data
        """
        unscaled_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
        
        # Unscale Reynolds number
        unscaled_batch['reynolds'] = batch['reynolds'] * self.scalers['reynolds']['std'] + self.scalers['reynolds']['mean']
        
        return unscaled_batch
    
    def save(self, path):
        """Save scaler statistics"""
        torch.save(self.scalers, path)
    
    def load(self, path):
        """Load scaler statistics"""
        self.scalers = torch.load(path)
        return self