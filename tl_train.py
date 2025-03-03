import torch
from torch.utils.data import DataLoader
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler
from experiment import ExperimentManager
from validation import ModelValidator, ValidationMetrics
from config import TrainingConfig
import logging
from pathlib import Path
from typing import Dict, Optional
import wandb
import argparse
from datetime import datetime
import logging

class ModelTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        dataloaders: Dict[str, DataLoader],
        device: torch.device,
        scaler: Optional[AirfoilDataScaler] = None,
        experiment: Optional[ExperimentManager] = None,
        validator: Optional[ModelValidator] = None
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.device = device
        self.scaler = scaler
        self.experiment = experiment
        self.validator = validator
        self.global_step = 0  # global steps for logging to wandb
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.dataloaders['train']):
            # Scale batch if scaler is provided
            if self.scaler:
                batch = self.scaler.transform(batch)
                
            # Move data to device
            airfoil_2d = batch['airfoil_2d'].to(self.device)
            geometry_3d = batch['geometry_3d'].to(self.device)
            pressure_3d = batch['pressure_3d'].to(self.device)
            mach = batch['mach'].to(self.device)
            reynolds = batch['reynolds'].to(self.device)
            z_coord = batch['z_coord'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(
                airfoil_2d,
                geometry_3d,
                mach,
                reynolds,
                z_coord
            )
            
            # Compute loss and backward pass
            loss = self.criterion(predictions, pressure_3d)
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.info(f'Epoch: {epoch} [{batch_idx}/{len(self.dataloaders["train"])}] '
                               f'Loss: {loss.item():.6f}')
                
            # Log to W&B if experiment manager is available
            if self.experiment:
                self.experiment.log_batch_metrics({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                }, self.global_step)
                self.global_step += 1
                
        return total_loss / len(self.dataloaders['train'])
    
    def run_validation(self, epoch: int, validation_type: str = 'val') -> ValidationMetrics:
        """Run validation on the specified dataset"""
        if self.validator is None:
            self.logger.warning("No validator provided. Skipping validation.")
            return None
            
        self.logger.info(f"Running {validation_type} validation for epoch {epoch}")
        return self.validator.validate_dataset(
            self.dataloaders[validation_type],
            self.global_step
        )
    
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch}")
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Periodic validation/Test
            metrics = {}
            if epoch % self.config.validation_freq == 0:
                # Validation set evaluation
                val_metrics = self.run_validation(epoch, 'val')
                if val_metrics:
                    metrics.update({
                        'val/loss': val_metrics.mse,
                        'val/rmse': val_metrics.rmse,
                        'val/mae': val_metrics.mae
                    })

                # Save best model
                if val_metrics.mse < best_val_loss:
                    best_val_loss = val_metrics.mse
                    if self.experiment:
                        self.experiment.save_checkpoint(
                            self.model,
                            self.optimizer,
                            epoch,
                            metrics,
                            self.scaler
                        )

                # Record training metrics
                train_metrics = self.run_validation(epoch, 'train')
                if train_metrics:
                    metrics.update({
                        'train/loss': train_metrics.mse,
                        'train/rmse': train_metrics.rmse,
                        'train/mae': train_metrics.mae
                    })
                
            # Test set evaluation if specified
            if hasattr(self.config, 'test_freq') and epoch % self.config.test_freq == 0:
                test_metrics = self.run_validation(epoch, 'test')
                if test_metrics:
                    metrics.update({
                        'test/loss': test_metrics.mse,
                        'test/rmse': test_metrics.rmse,
                        'test/mae': test_metrics.mae
                    })

                
            
            # Log epoch metrics
            metrics.update({'train/loss': train_loss, 'epoch': epoch})
            if self.experiment:
                self.experiment.log_epoch_metrics(metrics, self.global_step)
                
            # Regular model checkpoint
            if epoch % self.config.checkpoint_freq == 0 and self.experiment:
                self.experiment.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    metrics,
                    self.scaler
                )
                
            # Generate visualizations
            if epoch > 0 and epoch % self.config.viz_freq == 0 and self.experiment:
                self.experiment.log_model_predictions(
                    self.model,
                    'val',
                    self.dataloaders['val'],
                    self.device,
                    epoch,
                    #self.config.num_output_figs,
                    self.scaler,
                    self.global_step
                )
                
        self.logger.info("Training completed")

def train_model():
    """Main training function with improved configuration handling"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("main")

    # Load configuration from command line (with optional YAML base)
    config = TrainingConfig.from_args()

    # Set timestamp and initialize experiment directories and tracking
    experiment = ExperimentManager(config)
    experiment.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.setup_directories()
    experiment.setup_wandb()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Initialize or load scaler
    scaler = AirfoilDataScaler()
    scaler.fit(dataloaders['train'])
    
    # Initialize model and training components
    model = AirfoilTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = torch.nn.MSELoss()
    
    # Initialize validator
    validator = ModelValidator(
        model=model,
        criterion=criterion,
        device=device,
        scaler=scaler,
        log_to_wandb=True
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloaders=dataloaders,
        device=device,
        scaler=scaler,
        experiment=experiment,
        validator=validator
    )
    
    # Start training
    trainer.train()

    # Generate visualizations for last epoch (in the future exchange this for loading the best epoch)
    if experiment:
        for split in ['train', 'val', 'test']:
            experiment.log_model_predictions(
                model,
                split,
                dataloaders[split],
                device,
                config.num_epochs,
                scaler,
                None
            )
    
    # Cleanup
    if experiment:
        experiment.finish()

'''
# Evaluate loss
def evaluate(model, dataloader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    
    for batch in dataloader:

        # Scale batch
        batch = scaler.transform(batch)

        # Move data to device
        airfoil_2d = batch['airfoil_2d'].to(device)
        geometry_3d = batch['geometry_3d'].to(device)
        pressure_3d = batch['pressure_3d'].to(device)
        mach = batch['mach'].to(device)
        reynolds = batch['reynolds'].to(device)
        z_coord = batch['z_coord'].to(device)
        case_id = batch['case_id']
        
        # Forward pass
        predicted_pressures = model(
            airfoil_2d,
            geometry_3d,
            mach,
            reynolds,
            z_coord
        )
        
        # Compute loss
        loss = criterion(predicted_pressures, pressure_3d)
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_step(model, batch, optimizer, criterion, device, scaler):

    # Scale batch
    batch = scaler.transform(batch)

    # Move data to device
    airfoil_2d = batch['airfoil_2d'].to(device)
    geometry_3d = batch['geometry_3d'].to(device)
    pressure_3d = batch['pressure_3d'].to(device)
    mach = batch['mach'].to(device)
    reynolds = batch['reynolds'].to(device)
    z_coord = batch['z_coord'].to(device)
    case_id = batch['case_id']
    
    # Forward pass
    predicted_pressures = model(
        airfoil_2d,
        geometry_3d,
        mach,
        reynolds,
        z_coord
    )
    
    # Compute loss
    loss = criterion(predicted_pressures, pressure_3d)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# Modified training loop
def train_model(config_path):
    # Load configuration
    config = TrainingConfig.load(config_path)
    experiment = ExperimentManager(config)

    # Create dataloaders
    dataloaders = create_dataloaders(config.data_path, batch_size=config.batch_size)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Load or Create and fit the scaler
    scaler_fname = config.scaler_fname
    try:
        print(f"Loading scaler from {scaler_fname}")
        scaler = AirfoilDataScaler()
        scaler.load(scaler_fname)
    except FileNotFoundError:
        print(f"Fitting scaler and saving to {scaler_fname}")
        scaler = AirfoilDataScaler()
        scaler.fit(dataloaders['train'])  # Fit on training data only
        scaler.save(scaler_fname)
    
    # Your existing setup code here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AirfoilTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = torch.nn.MSELoss()
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_metrics = {'train_loss': 0.0}
        
        for batch_idx, batch in enumerate(train_loader):
            print(f'Epoch: {epoch} / Batch Idx {batch_idx}')
            # Your existing training step code here
            loss = train_step(model, batch, optimizer, criterion, device, scaler)
            
            # Log batch metrics
            #experiment.log_batch_metrics({
            #    'train_batch_loss': loss.item(),
            #    'learning_rate': optimizer.param_groups[0]['lr']
            #}, global_step)
            
            epoch_metrics['train_loss'] += loss.item()
            
        # Compute epoch metrics
        epoch_metrics['train_loss'] /= len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device, scaler)
        epoch_metrics['val_loss'] = val_loss
        
        # Log epoch metrics and visualizations
        experiment.log_epoch_metrics(epoch_metrics, epoch)
        experiment.log_model_predictions(model, val_loader, device, epoch, config.num_output_figs, scaler)
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint_freq == 0:
            experiment.save_checkpoint(model, optimizer, epoch, epoch_metrics)
    
    # Finish experiment
    experiment.finish()
'''

if __name__ == "__main__":
    # Train model
    train_model()

