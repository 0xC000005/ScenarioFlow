import random, time, os, json
import numpy as np
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm
from vae.base import BaseVAE
from vae.cvae_with_mem import CVAEMem
from vae.cvae import CVAE
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
from typing import Union

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(model: BaseVAE, dataloader):
    model.eval() # switch to eval model, will turn off randomness like dropout
    eval_losses = defaultdict(float)
    num_batches = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f"eval")):
        try:
            batch.to(model.device)
        except:
            pass

        losses = model.test_step(batch)

        for k, v in losses.items():
            eval_losses[k] += v.item()
        num_batches += 1

    for k, v in eval_losses.items():
        eval_losses[k] = v / num_batches

    return eval_losses
    

def train(model: BaseVAE, train_dataloader: DataLoader, valid_dataloader: DataLoader, 
          lr=1e-5, epochs=100, 
          model_dir="./", file_name="vanilla.pt"):
    model.train()
    optimizer = opt.AdamW(model.parameters(), lr)
    best_dev_loss = np.inf

    ## run for the specified number of epochs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if "." in file_name:
        file_prefix = file_name.split(".")[0]
    else:
        file_prefix = file_name
    log_file = open(f"{model_dir}/{file_prefix}-{epochs}-log.txt", "w", encoding="utf-8")

    print("Model config: ", file=log_file)
    print(json.dumps(model.config, indent=True), file=log_file)
    print(f"LR: {lr}", file=log_file)
    print(f"Epochs: {epochs}", file=log_file)
    print("", file=log_file)
    log_file.flush()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
            try:
                batch.to(model.device)
            except:
                pass

            losses = model.train_step(batch, optimizer)

            for k, v in losses.items():
                train_losses[k] += v.item()
            num_batches += 1
        for k, v in train_losses.items():
            train_losses[k] = v / (num_batches)
        
        dev_losses = model_eval(model, valid_dataloader)

        if dev_losses["loss"] < best_dev_loss:
            best_dev_loss = dev_losses["loss"]
            model.save_weights(optimizer, model_dir, file_prefix)

        formatted_train_loss = ", ".join([f'{k}: {v:.3f}' for k, v in train_losses.items()])
        formatted_dev_loss = ", ".join([f'{k}: {v:.3f}' for k, v in dev_losses.items()])
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}")
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}", file=log_file)
    print(f"training finished, total time :: {time.time() - start_time}")
    print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
    log_file.close()
    return train_losses, dev_losses

def train_with_scheduled_sampling(
    model: BaseVAE,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    lr=1e-5,
    epochs=400,
    model_dir="./",
    file_name="backfill_model.pt",
    teacher_forcing_epochs=200,
    horizons=[1, 7, 14, 30]
):
    """
    Two-phase training with scheduled sampling for backfilling.

    Phase 1 (epochs 0 to teacher_forcing_epochs):
        - Standard single-step teacher forcing using train_step()
        - Model learns to predict 1 day ahead with ground truth context
        - Builds strong short-term prediction capability

    Phase 2 (epochs teacher_forcing_epochs+1 to epochs):
        - Multi-horizon training using train_step_multihorizon()
        - Model learns to predict multiple horizons [1, 7, 14, 30] simultaneously
        - Improves long-term forecasting without forgetting short-term

    This approach follows the BACKFILL_MVP_PLAN.md Phase 2.2 design:
    - Gradual transition from teacher forcing to multi-horizon
    - Prevents catastrophic forgetting of short-term predictions
    - Improves stability compared to multi-horizon from scratch

    Args:
        model: CVAEMemRand instance (must have train_step_multihorizon method)
        train_dataloader: Training data (must have T >= 5 + 30 = 35 for max horizon)
        valid_dataloader: Validation data
        lr: Learning rate (default 1e-5)
        epochs: Total training epochs (default 400)
        model_dir: Directory to save model checkpoints
        file_name: Model filename (will add .pt extension)
        teacher_forcing_epochs: When to switch to multi-horizon (default 200)
        horizons: List of horizons for Phase 2 (default [1, 7, 14, 30])

    Returns:
        Tuple of (train_losses, dev_losses) from last epoch
    """
    model.train()
    optimizer = opt.AdamW(model.parameters(), lr)
    best_dev_loss = np.inf

    # Setup logging
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if "." in file_name:
        file_prefix = file_name.split(".")[0]
    else:
        file_prefix = file_name
    log_file = open(f"{model_dir}/{file_prefix}-{epochs}-log.txt", "w", encoding="utf-8")

    print("=" * 80, file=log_file)
    print("SCHEDULED SAMPLING TRAINING", file=log_file)
    print("=" * 80, file=log_file)
    print("Model config: ", file=log_file)
    print(json.dumps(model.config, indent=True), file=log_file)
    print(f"LR: {lr}", file=log_file)
    print(f"Total Epochs: {epochs}", file=log_file)
    print(f"Teacher Forcing Epochs: {teacher_forcing_epochs} (Phase 1)", file=log_file)
    print(f"Multi-Horizon Epochs: {epochs - teacher_forcing_epochs} (Phase 2)", file=log_file)
    print(f"Multi-Horizon Targets: {horizons}", file=log_file)
    print("=" * 80, file=log_file)
    print("", file=log_file)
    log_file.flush()

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0

        # Determine training mode for this epoch
        if epoch < teacher_forcing_epochs:
            # Phase 1: Teacher Forcing (single-step)
            mode_str = f"Phase 1 - Teacher Forcing (epoch {epoch+1}/{teacher_forcing_epochs})"
            print(f"\n{mode_str}")

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
                try:
                    batch.to(model.device)
                except:
                    pass

                # Standard single-step training
                losses = model.train_step(batch, optimizer)

                for k, v in losses.items():
                    train_losses[k] += v.item()
                num_batches += 1

        else:
            # Phase 2: Multi-Horizon Training
            mode_str = f"Phase 2 - Multi-Horizon {horizons} (epoch {epoch+1-teacher_forcing_epochs}/{epochs-teacher_forcing_epochs})"
            print(f"\n{mode_str}")

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
                try:
                    batch.to(model.device)
                except:
                    pass

                # Multi-horizon training
                losses = model.train_step_multihorizon(batch, optimizer, horizons=horizons)

                for k, v in losses.items():
                    if isinstance(v, dict):
                        # horizon_losses is a dict, log each separately
                        for h, loss_val in v.items():
                            train_losses[f"{k}_h{h}"] += loss_val
                    else:
                        train_losses[k] += v if isinstance(v, float) else v.item()
                num_batches += 1

        # Average training losses
        for k, v in train_losses.items():
            train_losses[k] = v / num_batches

        # Validation (use single-step for consistency)
        dev_losses = model_eval(model, valid_dataloader)

        # Save best model based on validation loss
        if dev_losses["loss"] < best_dev_loss:
            best_dev_loss = dev_losses["loss"]
            model.save_weights(optimizer, model_dir, file_prefix)
            save_marker = "✓ BEST"
        else:
            save_marker = ""

        # Format and print losses
        formatted_train_loss = ", ".join([f'{k}: {v:.3f}' for k, v in train_losses.items()])
        formatted_dev_loss = ", ".join([f'{k}: {v:.3f}' for k, v in dev_losses.items()])
        epoch_time = time.time() - epoch_start_time

        print(f"{mode_str}")
        print(f"train loss :: {formatted_train_loss}")
        print(f"dev loss :: {formatted_dev_loss}")
        print(f"time :: {epoch_time:.1f}s {save_marker}")

        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {epoch_time} {save_marker}", file=log_file)
        log_file.flush()

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Training finished! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation loss: {best_dev_loss:.6f}")
    print(f"{'=' * 80}")

    print(f"\ntraining finished, total time :: {total_time}", file=log_file)
    print(f"best validation loss :: {best_dev_loss}", file=log_file)
    log_file.close()

    return train_losses, dev_losses

def test(model: BaseVAE, valid_dataloader: DataLoader, test_dataloader: DataLoader, model_fn="./vanilla"):
    model.load_weights(f=model_fn)
    dev_losses = model_eval(model, valid_dataloader)
    test_losses = model_eval(model, test_dataloader)

    formatted_dev_loss = ", ".join([f'{k}: {v:.3f}' for k, v in dev_losses.items()])
    formatted_test_loss = ", ".join([f'{k}: {v:.3f}' for k, v in test_losses.items()])
    print(f"dev loss: {formatted_dev_loss}, \ntest_loss: {formatted_test_loss}")
    return dev_losses, test_losses

def plot_surface(original_data: np.ndarray, vae_output: np.ndarray):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = [0.7,0.85,1,1.15,1.3]
    y = [0.08333,0.25,0.5,1,2]
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, original_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_surface(x, y, vae_output, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("K/S")
    ax.set_ylabel("ttm")
    plt.show()

def plot_surface_separate(original_data: np.ndarray, vae_output: np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d"})
    x = [0.7,0.85,1,1.15,1.3]
    y = [0.08333,0.25,0.5,1,2]
    x, y = np.meshgrid(x, y)

    ax[0].plot_surface(x, y, original_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax[0].set_xlabel("K/S")
    ax[0].set_ylabel("ttm")
    ax[0].set_title("Original surface")
    
    ax[1].plot_surface(x, y, vae_output, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax[1].set_xlabel("K/S")
    ax[1].set_ylabel("ttm")
    ax[1].set_title("VAE surface")
    
    ax[2].plot_surface(x, y, original_data, cmap=cm.Blues, linewidth=0, antialiased=False)
    ax[2].plot_surface(x, y, vae_output, cmap=cm.Reds, linewidth=0, antialiased=False)
    ax[2].set_xlabel("K/S")
    ax[2].set_ylabel("ttm")
    ax[2].set_title("Both surfaces")
    
    plt.subplots_adjust(right=2.0, wspace=0.3)
    plt.show()

def generate_surface_path(surf_data, ex_data, model_data, path_idx=8000, model_type: Union[CVAE, CVAEMem] = CVAEMem):
    '''
        This function is for SABR data. Also this function applies an aggregated approach. 
        i.e. generate new surfaces based on previously generated data
    '''
    model_config = model_data["model_config"]
    model = model_type(model_config)
    model.load_weights(dict_to_load=model_data)
    seq_len = model_config["seq_len"]
    if "ex_feats_dim" in model_config:
        use_ex_feats = model_config["ex_feats_dim"] > 0
    else:
        use_ex_feats = False
    all_simulation = {
        "surface": [surf_data[path_idx][i] for i in range(seq_len-1)],
        "ex_feats": [ex_data[path_idx][i] for i in range(seq_len-1)] if use_ex_feats else None,
    }
    steps_to_sim = len(surf_data[path_idx]) + 1 - seq_len
    for i in range(steps_to_sim):
        ctx_data = {
            "surface": torch.from_numpy(np.array(all_simulation["surface"][i:(i+seq_len-1)])), 
            "ex_feats": torch.from_numpy(np.array(all_simulation["ex_feats"][i:(i+seq_len-1)])).unsqueeze(-1) if use_ex_feats else None
        }
        if use_ex_feats:
            surf, ex_feat = model.get_surface_given_conditions(ctx_data) 
        else:
            ctx_data.pop("ex_feats")
            surf = model.get_surface_given_conditions(ctx_data) 
        surf = surf.detach().cpu().numpy().reshape((5,5))
        all_simulation["surface"].append(surf)
        if use_ex_feats:
            ex_feat = ex_feat.detach().cpu().numpy().reshape((1,))[0]
            all_simulation["ex_feats"].append(ex_feat)

    all_simulation["surface"] = np.array(all_simulation["surface"])
    if use_ex_feats:
        all_simulation["ex_feats"] = np.array(all_simulation["ex_feats"])
    
    return all_simulation

def generate_surface_spx(surf_data, ex_data, model_data, start_time=5000, steps_to_sim=30, model_type: Union[CVAE, CVAEMem] = CVAEMem):
    '''
        This function is for S&P500 data. Also this function applies an aggregated approach. 
        i.e. generate new surfaces based on previously generated data

        This function uses the original return/price as ex features for simulation
    '''
    model_config = model_data["model_config"]
    model = model_type(model_config)
    model.load_weights(dict_to_load=model_data)
    seq_len = model_config["seq_len"]
    ctx_len = model_config["ctx_len"]
    if "ex_feats_dim" in model_config:
        use_ex_feats = model_config["ex_feats_dim"] > 0
    else:
        use_ex_feats = False
    all_simulation = {
        "surface": [surf_data[start_time+i] for i in range(ctx_len)],
        "ex_feats": [ex_data[start_time+i] for i in range(steps_to_sim + ctx_len)] if use_ex_feats else None,
    }
    for i in range(steps_to_sim):
        ctx_data = {
            "surface": torch.from_numpy(np.array(all_simulation["surface"][i:(i+ctx_len)])), 
            "ex_feats": torch.from_numpy(np.array(all_simulation["ex_feats"][i:(i+ctx_len)])).unsqueeze(-1) if use_ex_feats else None
        }
        if use_ex_feats:
            surf, _ = model.get_surface_given_conditions(ctx_data) 
        else:
            ctx_data.pop("ex_feats")
            surf = model.get_surface_given_conditions(ctx_data) 
        surf = surf.detach().cpu().numpy().reshape((5,5))
        all_simulation["surface"].append(surf)
        # if use_ex_feats:
        #     ex_feat = ex_feat.detach().cpu().numpy().reshape((1,))[0]
        #     all_simulation["ex_feats"].append(ex_feat)

    all_simulation["surface"] = np.array(all_simulation["surface"])
    if use_ex_feats:
        all_simulation["ex_feats"] = np.array(all_simulation["ex_feats"])
    
    return all_simulation

def plot_surface_time_series(vae_output, title_label="VAE"):
    if isinstance(vae_output, dict):
        surfaces = vae_output["surface"]
        ex_feats = vae_output["ex_feats"]
    else:
        surfaces = vae_output
        ex_feats = None
    nrows = surfaces.shape[0] // 5
    if surfaces.shape[0] % 5 != 0:
        nrows += 1
    fig, ax = plt.subplots(nrows=nrows, ncols=5, subplot_kw={"projection": "3d"})
    x = [0.7,0.85,1,1.15,1.3]
    y = [0.08333,0.25,0.5,1,2]
    x, y = np.meshgrid(x, y)

    for i in range(len(surfaces)):
        r = i // 5
        c = i % 5
        ax[r][c].plot_surface(x, y, surfaces[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax[r][c].set_xlabel("K/S")
        ax[r][c].set_ylabel("ttm")
        if ex_feats is not None:
            ax[r][c].set_title(f"{title_label} surface on day {i},\nex_feat={ex_feats[i]:.4f}")
        else:
            ax[r][c].set_title(f"{title_label} surface on day {i}")
    
    plt.subplots_adjust(right=4.0, top=4.0, hspace=0.5)
    plt.show()


# ==============================================================================
# Multi-Offset Autoregressive Training Functions
# Added for Context=60 4-phase curriculum training
# ==============================================================================

def train_autoregressive_step(
    model, batch, optimizer,
    ar_steps, horizon, offset
):
    """
    Single autoregressive training step with specified offset.

    Key enhancement from existing implementations:
    - Adds 'offset' parameter to control context window sliding
    - Handles both overlapping (offset < horizon) and non-overlapping (offset = horizon)
    - Context window properly combines real and generated data

    Args:
        model: CVAEMemRand instance
        batch: dict with "surface" (B, T, H, W) and optional "ex_feats"
        optimizer: PyTorch optimizer
        ar_steps: Number of autoregressive steps (typically 3)
        horizon: Prediction horizon (60 or 90 days)
        offset: Context window shift amount (30, 45, 60, or 90 days)

    Returns:
        dict with loss metrics including ar_steps and offset used
    """
    optimizer.zero_grad()

    surface = batch["surface"].to(model.device)
    B, T, H, W = surface.shape
    C = model.config.get("context_len", 60)

    # Validate sequence length
    required_len = C + (ar_steps * offset)
    if T < required_len:
        raise ValueError(
            f"Sequence too short for offset={offset}, ar_steps={ar_steps}: "
            f"need {required_len} days, got {T} days"
        )

    # Initialize with real context
    context = {"surface": surface[:, :C, :, :]}
    if "ex_feats" in batch:
        context["ex_feats"] = batch["ex_feats"][:, :C, :].to(model.device)

    total_loss = 0
    total_recon = 0
    total_kl = 0
    predictions = []  # Store predictions for context updates

    # Autoregressive rollout
    for step in range(ar_steps):
        # Temporarily set model horizon
        original_horizon = model.horizon
        model.horizon = horizon

        # Ground truth for this horizon
        target_start = C + (step * offset)
        target_end = target_start + horizon
        target_surface = surface[:, target_start:target_end, :, :]

        # Create input with context + target (model expects both)
        model_input = {
            "surface": torch.cat([context["surface"], target_surface], dim=1)
        }
        if "ex_feats" in context:
            target_ex_feats = batch["ex_feats"][:, target_start:target_end, :].to(model.device)
            model_input["ex_feats"] = torch.cat([context["ex_feats"], target_ex_feats], dim=1)

        # Forward pass
        if "ex_feats" in model_input:
            surf_recon, ex_recon, z_mean, z_log_var, z = model(model_input)
        else:
            surf_recon, z_mean, z_log_var, z = model(model_input)

        model.horizon = original_horizon

        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(surf_recon, target_surface)

        # Handle ex_feats if present
        if "ex_feats" in model_input:
            if model.config.get("ex_loss_on_ret_only", False):
                # ex_recon shape: (B, horizon, ex_feats_dim)
                ex_recon_loss = ex_recon[:, :, :1]
                target_ex_loss = target_ex_feats[:, :, :1]
            else:
                ex_recon_loss = ex_recon
                target_ex_loss = target_ex_feats
            ex_loss = model.ex_feats_loss_fn(ex_recon_loss, target_ex_loss)
            recon_loss = recon_loss + model.config["re_feat_weight"] * ex_loss

        # KL divergence
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        # Step loss
        step_loss = recon_loss + model.config["kl_weight"] * kl_loss
        total_loss += step_loss
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        # Store prediction
        pred_surface = surf_recon  # (B, horizon, H, W)
        predictions.append(pred_surface)

        # Update context for next step (if not last step)
        if step < ar_steps - 1:
            # New context window starts at this global position
            new_context_start = (step + 1) * offset

            # Build new context: [new_context_start : new_context_start + C]
            # This may include real data and/or predictions

            if new_context_start + C <= C:
                # Still entirely within real data
                context["surface"] = surface[:, new_context_start:new_context_start + C, :, :]
                if "ex_feats" in batch:
                    context["ex_feats"] = batch["ex_feats"][:, new_context_start:new_context_start + C, :].to(model.device)

            elif new_context_start < C:
                # Mixed: some real data + some predictions
                real_end = C
                real_portion = surface[:, new_context_start:real_end, :, :]

                # Collect prediction portions that overlap with [new_context_start, new_context_start + C]
                pred_portions = []
                for prev_step, pred in enumerate(predictions):
                    pred_start_global = C + prev_step * offset
                    pred_end_global = pred_start_global + horizon

                    # Check overlap
                    overlap_start = max(pred_start_global, new_context_start)
                    overlap_end = min(pred_end_global, new_context_start + C)

                    if overlap_start < overlap_end:
                        # Extract relevant portion from prediction
                        local_start = overlap_start - pred_start_global
                        local_end = overlap_end - pred_start_global
                        pred_portions.append(pred[:, local_start:local_end, :, :])

                # Concatenate real and predicted portions
                if pred_portions:
                    pred_concat = torch.cat(pred_portions, dim=1)
                    context["surface"] = torch.cat([real_portion, pred_concat], dim=1)
                else:
                    context["surface"] = real_portion

                # Handle ex_feats similarly
                if "ex_feats" in batch:
                    # Note: We don't have predicted ex_feats stored, so we only use real portion
                    # This is acceptable since ex_feats are auxiliary and we mainly care about surface
                    real_ex_feats = batch["ex_feats"][:, new_context_start:real_end, :].to(model.device)
                    # Pad or truncate to match context length
                    current_len = real_ex_feats.shape[1]
                    if current_len < C:
                        # Need to pad - use last real value
                        pad_len = C - current_len
                        last_val = real_ex_feats[:, -1:, :].repeat(1, pad_len, 1)
                        context["ex_feats"] = torch.cat([real_ex_feats, last_val], dim=1)
                    else:
                        context["ex_feats"] = real_ex_feats[:, :C, :]

            else:
                # Only predictions (no real data)
                pred_portions = []
                for prev_step, pred in enumerate(predictions):
                    pred_start_global = C + prev_step * offset
                    pred_end_global = pred_start_global + horizon

                    overlap_start = max(pred_start_global, new_context_start)
                    overlap_end = min(pred_end_global, new_context_start + C)

                    if overlap_start < overlap_end:
                        local_start = overlap_start - pred_start_global
                        local_end = overlap_end - pred_start_global
                        pred_portions.append(pred[:, local_start:local_end, :, :])

                # Take last C days from predictions
                context["surface"] = torch.cat(pred_portions, dim=1)[:, -C:, :, :]

                # Handle ex_feats
                if "ex_feats" in batch:
                    # Use last real ex_feats value and repeat for context length
                    # This is a simplification since we don't store predicted ex_feats
                    last_real_idx = min(C - 1, batch["ex_feats"].shape[1] - 1)
                    last_ex_feats = batch["ex_feats"][:, last_real_idx:last_real_idx+1, :].to(model.device)
                    context["ex_feats"] = last_ex_feats.repeat(1, C, 1)

    # Average loss over steps
    total_loss = total_loss / ar_steps
    total_loss.backward()
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "reconstruction_loss": total_recon / ar_steps,
        "kl_loss": total_kl / ar_steps,
        "ar_steps": ar_steps,
        "offset": offset,
    }


def train_autoregressive_multi_offset(
    model, batch, optimizer,
    horizon, offsets, ar_steps=3
):
    """
    Train with multiple offset strategies (random sampling).

    This function implements multi-offset training by randomly sampling
    an offset value for each batch. This builds model robustness to
    different context overlap strategies.

    Example usage in Phase 3:
        horizon=60, offsets=[30, 60]
        - 50% of batches train with offset=30 (50% overlap)
        - 50% of batches train with offset=60 (non-overlapping)

    Example usage in Phase 4:
        horizon=90, offsets=[45, 90]
        - 50% of batches train with offset=45 (50% overlap, ensemble-ready)
        - 50% of batches train with offset=90 (deployment target)

    Args:
        model: CVAEMemRand instance
        batch: Training batch from dataloader
        optimizer: PyTorch optimizer
        horizon: Prediction horizon (60 or 90 days)
        offsets: List of offset values to sample from (e.g., [30, 60] or [45, 90])
        ar_steps: Number of AR steps (default: 3)

    Returns:
        dict with loss metrics including the sampled offset
    """
    # Randomly sample offset for this batch
    offset = random.choice(offsets)

    # Run autoregressive training with sampled offset
    return train_autoregressive_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        ar_steps=ar_steps,
        horizon=horizon,
        offset=offset
    )