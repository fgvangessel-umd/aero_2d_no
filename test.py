import torch
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler
from experiment import ExperimentManager
from tl_viz import plot_3d_wing_predictions
from utils import load_checkpoint, calculate_airfoil_forces, select_case
from config import TrainingConfig
from typing import Dict, Optional
import argparse
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from typing import Tuple
import sys
from matplotlib import pyplot as plt
from validation import ModelValidator
import pickle

if __name__ == "__main__":
    """Main testing function"""
    # Load config file info
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    config = TrainingConfig.load(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    data_path = 'data/data_standard'
    dataloaders = create_dataloaders(
        data_path,
        batch_size=10000,
        num_workers=config.num_workers
    )
    
    # Initialize or load scaler
    scaler = AirfoilDataScaler()
    
    # Initialize model and training components
    model = AirfoilTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = torch.nn.MSELoss()

    # Set checkpoint path
    checkpoint_path = 'experiments/baseline_transformer_standard_20250218_091213/models/checkpoint_epoch_917.pt'

    # Load checkpointed model
    model, optimizer, scaler, epoch, metrics = load_checkpoint(checkpoint_path, model, optimizer, scaler)

    # Initialize validator
    validator = ModelValidator(
        model=model,
        criterion=criterion,
        device=device,
        scaler=scaler,
        log_to_wandb=False
    )

    model.eval()

    with torch.no_grad():

        # Get val cases
        for batch_idx, batch in enumerate(dataloaders['val']):
            # Move data to device
            val_cases = batch['case_id'].numpy().tolist()
        val_cases = list(set(val_cases))

        # Get test cases
        for batch_idx, batch in enumerate(dataloaders['test']):
            # Move data to device
            test_cases = batch['case_id'].numpy().tolist()
        test_cases = list(set(test_cases))

        case_ids = {'val': val_cases, 'test': test_cases}
        true_lift = []
        pred_lift = []
        true_drag = []
        pred_drag = []

        # Load data
        for split in ['val', 'test']:
            for case_id in case_ids[split]:
                for batch_idx, batch in enumerate(dataloaders[split]):
                
                    true_reynolds = batch['reynolds'].to(device)
                    batch = scaler.transform(batch)
                        
                    airfoil_2d, geometry_3d, pressure_3d, mach, reynolds, true_reynolds, z_coord = \
                        select_case(batch, true_reynolds, case_id, device)
                    
                    # Make model predictions
                    predictions = model(
                        airfoil_2d,
                        geometry_3d,
                        mach,
                        reynolds,
                        z_coord
                    )

                    # Convert data formats
                    xy_2d = airfoil_2d[:,:,1:3].cpu().numpy()
                    xy_3d = geometry_3d[:,:,1:3].cpu().numpy()
                    z_coord = z_coord.cpu().numpy()
                    p_2d = airfoil_2d[:,:,3].cpu().numpy()
                    p_3d_true = pressure_3d.cpu().numpy()
                    p_3d_pred = predictions.cpu().numpy()
                    case_data = {'case_id': case_id, 'mach': mach[0].item(), 'reynolds':true_reynolds[0].item()}
                    fname = f'model_test/mach/predictions_{case_id}.png'

                    plot_3d_wing_predictions(xy_2d, xy_3d, p_2d, p_3d_true, p_3d_pred, z_coord, case_data, fname)
                    
                    force_metrics = validator.compute_force_metrics(predictions, pressure_3d, geometry_3d[:,:,1:3], alpha=2.5)
                    
                    true_lift.append(force_metrics['true_wing_lift'])
                    pred_lift.append(force_metrics['pred_wing_lift'])
                    true_drag.append(force_metrics['true_wing_drag'])
                    pred_drag.append(force_metrics['pred_wing_drag'])
            

        lift_drag_dict = {
            'true_lift': true_lift,
            'true_lift': true_drag,
            'true_lift': pred_lift,
            'true_lift': pred_drag
            }

        # Save the dictionary to a pickle file
        with open('lift_drag_dict.pickle', 'wb') as handle:
            pickle.dump(lift_drag_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,10), tight_layout=True)
        axs[0].scatter(true_lift, pred_lift)
        axs[1].scatter(true_drag, pred_drag)
        axs[0].plot(np.linspace(np.min(true_lift), np.max(true_lift)), np.linspace(np.min(true_lift), np.max(true_lift)), c='k')
        axs[1].plot(np.linspace(np.min(true_drag), np.max(true_drag)), np.linspace(np.min(true_drag), np.max(true_drag)), c='k')
        plt.savefig('lift_drag_parity.png')

        corr_lift, _ = stats.pearsonr(true_lift, pred_lift)
        corr_drag, _ = stats.pearsonr(true_drag, pred_drag)

        r2_lift = r2_score(true_lift, pred_lift)
        r2_drag = r2_score(true_drag, pred_drag)

        mae_lift = mean_absolute_error(true_lift, pred_lift)
        mae_drag = mean_absolute_error(true_drag, pred_drag)

        mape_lift = mean_absolute_percentage_error(true_lift, pred_lift)
        mape_drag = mean_absolute_percentage_error(true_drag, pred_drag)

        print(f'Correlation Lift: {corr_lift: .2e}')
        print(f'Correlation Drag: {corr_drag: .2e}\n')

        print(f'R2 Lift: {r2_lift: .2e}')
        print(f'R2 Drag: {r2_drag: .2e}\n')

        print(f'MAE Lift: {mae_lift: .2e}')
        print(f'MAE Drag: {mae_drag: .2e}\n')

        print(f'MAPE Lift: {mape_lift: .2e}')
        print(f'MAPE Drag: {mape_drag: .2e}\n')