from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.backfill.block_ar.evaluate_213a_h1_conditional_distribution_suite import (
    _get_loader as _get_h1_loader,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


@dataclass
class RolloutBatch:
    history_01: torch.Tensor
    history_norm: torch.Tensor
    future_01: torch.Tensor
    future_norm: torch.Tensor


class HistoryFutureDictDataset(Dataset):
    def __init__(self, history_norm: torch.Tensor, future_norm: torch.Tensor):
        self.history_norm = history_norm
        self.future_norm = future_norm

    def __len__(self) -> int:
        return int(self.history_norm.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.history_norm[idx],
            "future": self.future_norm[idx],
        }


class OneDayKernelRolloutWrapper:
    """Adapt a one-day `sample_next_iv` kernel to a multi-day `sample_batched` API."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.noise_dim = getattr(model, "noise_dim", None)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    @staticmethod
    def _coerce_next_iv(samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim == 3 and samples.shape[-1] == 25:
            return samples.view(samples.shape[0], samples.shape[1], 5, 5)
        if samples.ndim == 4 and samples.shape[-2:] == (5, 5):
            return samples
        raise ValueError(f"Unexpected sample_next_iv shape: {tuple(samples.shape)}")

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if history_is_normalized:
            history_01 = denormalize_iv(history)
        else:
            history_01 = history
        batch_size, hist_len = history_01.shape[:2]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))

        chunks: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            hist_k = history_01.unsqueeze(1).expand(batch_size, k, hist_len, 5, 5)
            hist_k = hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()

            frames: list[torch.Tensor] = []
            for _ in range(n_steps):
                next_iv = self.model.sample_next_iv(hist_k, n_samples=1)
                next_iv = self._coerce_next_iv(next_iv).squeeze(1)
                frames.append(next_iv.view(batch_size, k, 5, 5))
                hist_k = torch.cat(
                    [hist_k[:, 1:], next_iv.view(batch_size * k, 1, 5, 5)],
                    dim=1,
                )
            chunks.append(torch.stack(frames, dim=2))
        return torch.cat(chunks, dim=1)


def load_one_day_kernel(
    model_type: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if model_type == "220c":
        from experiments.backfill.block_ar.train_220c_rollout_finetune_212ai import (
            load_model as load_220c_model,
        )

        return load_220c_model(checkpoint_path, device)
    if model_type in {"220d", "220e", "220f"}:
        from experiments.backfill.block_ar.train_220d_recurrent_flow_transition import (
            load_model as load_220d_model,
        )

        return load_220d_model(checkpoint_path, device)
    if model_type == "221a":
        from experiments.backfill.block_ar.train_220d_recurrent_flow_transition import (
            RecurrentFlowTransitionModel,
        )

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = payload["config"]
        model = RecurrentFlowTransitionModel(
            n_cells=cfg["n_cells"],
            history_feat_dim=cfg["history_feat_dim"],
            hidden_dim=cfg["hidden_dim"],
            gru_layers=cfg["gru_layers"],
            gru_dropout=cfg["gru_dropout"],
            flow_hidden=cfg["flow_hidden"],
            n_coupling_layers=cfg["n_coupling_layers"],
            ewma_alpha=cfg["ewma_alpha"],
            scale_floor=cfg["scale_floor"],
            include_scale_feature=cfg["include_scale_feature"],
            support_lo=cfg.get("support_lo", 0.01),
            support_hi=cfg.get("support_hi", 1.0),
        )
        model.load_state_dict(payload["model_state_dict"], strict=False)
        model.init_recurrent_cells_from_gru()
        model.to(device).eval()
        return model, payload
    if model_type in {"221d", "221e"}:
        from experiments.backfill.block_ar.train_221d_adapter_multiday_ar_conditional_flow import (
            AdapterRecurrentFlowTransitionModel,
        )

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = payload["config"]
        model = AdapterRecurrentFlowTransitionModel(
            n_cells=cfg["n_cells"],
            history_feat_dim=cfg["history_feat_dim"],
            hidden_dim=cfg["hidden_dim"],
            gru_layers=cfg["gru_layers"],
            gru_dropout=cfg["gru_dropout"],
            flow_hidden=cfg["flow_hidden"],
            n_coupling_layers=cfg["n_coupling_layers"],
            ewma_alpha=cfg["ewma_alpha"],
            scale_floor=cfg["scale_floor"],
            include_scale_feature=cfg["include_scale_feature"],
            support_lo=cfg.get("support_lo", 0.01),
            support_hi=cfg.get("support_hi", 1.0),
            adapter_hidden=cfg.get("adapter_hidden", 64),
            adapter_ramp_steps=cfg.get("adapter_ramp_steps", 0),
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device).eval()
        return model, payload
    if model_type in {"221b", "221c"}:
        from experiments.backfill.block_ar.train_220d_recurrent_flow_transition import (
            RecurrentFlowTransitionModel,
        )

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = payload["config"]
        model = RecurrentFlowTransitionModel(
            n_cells=cfg["n_cells"],
            history_feat_dim=cfg["history_feat_dim"],
            hidden_dim=cfg["hidden_dim"],
            gru_layers=cfg["gru_layers"],
            gru_dropout=cfg["gru_dropout"],
            flow_hidden=cfg["flow_hidden"],
            n_coupling_layers=cfg["n_coupling_layers"],
            ewma_alpha=cfg["ewma_alpha"],
            scale_floor=cfg["scale_floor"],
            include_scale_feature=cfg["include_scale_feature"],
            support_lo=cfg.get("support_lo", 0.01),
            support_hi=cfg.get("support_hi", 1.0),
        )
        # DO NOT call init_recurrent_cells_from_gru -- cells are trained
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device).eval()
        return model, payload
    if model_type == "220g":
        from experiments.backfill.block_ar.train_220g_slow_regime_flow_transition import (
            load_model as load_220g_model,
        )

        return load_220g_model(checkpoint_path, device)
    if model_type == "220i":
        from experiments.backfill.block_ar.train_220i_slow_factor_ema_flow_transition import (
            load_model as load_220i_model,
        )

        return load_220i_model(checkpoint_path, device)
    if model_type == "220j":
        from experiments.backfill.block_ar.train_220j_slow_factor_drift_flow_transition import (
            load_model as load_220j_model,
        )

        return load_220j_model(checkpoint_path, device)
    if model_type == "183c":
        from experiments.backfill.block_ar.analyze_183c_best_mechanism import (
            load_model as load_183c_model,
        )

        return load_183c_model(checkpoint_path, device)
    if model_type in {"223a", "223b", "224a", "224b", "224c", "224d", "224e"}:
        from experiments.backfill.block_ar.train_223a_generated_history_finetune import (
            load_model as load_223_model,
        )

        return load_223_model(checkpoint_path, device)
    if model_type in {"226a", "226b"}:
        from experiments.backfill.block_ar.train_226a_factor_decoupled_flow import (
            load_model as load_226a_model,
        )

        return load_226a_model(checkpoint_path, device)
    if model_type == "227a":
        from experiments.backfill.block_ar.train_227a_factor_ar import (
            load_model as load_227a_model,
        )

        return load_227a_model(checkpoint_path, device)
    if model_type in {"231a", "231b", "231c"}:
        from experiments.backfill.block_ar.train_231a_hybrid_recurrent_factor_flow import (
            load_model as load_231a_model,
        )

        return load_231a_model(checkpoint_path, device)
    if model_type == "232a":
        from experiments.backfill.block_ar.train_232a_regime_mixture import (
            load_model as load_232a_model,
        )

        return load_232a_model(checkpoint_path, device)
    if model_type == "232b":
        from experiments.backfill.block_ar.train_232b_heavy_tail import (
            load_model as load_232b_model,
        )

        return load_232b_model(checkpoint_path, device)
    if model_type in {"232c", "232d"}:
        # 232c/d use 227a's FactorARModel architecturally (loss-only variants)
        from experiments.backfill.block_ar.train_227a_factor_ar import (
            load_model as load_227a_model,
        )

        return load_227a_model(checkpoint_path, device)
    if model_type in {"233a", "233a_full", "233a_B", "233a_C"}:
        from experiments.backfill.block_ar.train_233a_twopath_factor_ar import (
            load_model as load_233a_model,
        )

        return load_233a_model(checkpoint_path, device)
    if model_type in {"233a_v1_2", "233a_v1_2_full"}:
        from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import (
            load_model as load_233a_v1_2_model,
        )

        return load_233a_v1_2_model(checkpoint_path, device)
    if model_type in {"240a", "240a_joint_chunk"}:
        from experiments.backfill.block_ar.train_240a_joint_chunk_flow import (
            load_model as load_240a_model,
        )

        return load_240a_model(checkpoint_path, device)
    if model_type in {"240b", "240b_joint_chunk_crps"}:
        from experiments.backfill.block_ar.train_240b_joint_chunk_crps import (
            load_model as load_240b_model,
        )

        return load_240b_model(checkpoint_path, device)
    if model_type in {"240c", "240c_iter", "240c_iter_ddim_crps"}:
        from experiments.backfill.block_ar.train_240c_iter_ddim_crps import (
            load_model as load_240c_model,
        )

        return load_240c_model(checkpoint_path, device)
    if model_type in {
        "250a", "250b", "250c",
        "251a", "251b", "251c", "251d", "251e", "251f", "251g", "251h",
    }:
        from diffusion.block_ar.neural_factor import load_model as load_250_model

        return load_250_model(checkpoint_path, device)
    if model_type == "252a":
        from diffusion.block_ar.mean_first_factor import load_model as load_252a_model

        return load_252a_model(checkpoint_path, device)
    if model_type == "253b":
        from diffusion.block_ar.dynamic_change_factor_shock_ssm import load_model as load_253b_model

        return load_253b_model(checkpoint_path, device)
    if model_type == "253c":
        from diffusion.block_ar.dynamic_change_factor_selective_pulse_ssm import (
            load_model as load_253c_model,
        )

        return load_253c_model(checkpoint_path, device)
    if model_type == "254a":
        from diffusion.block_ar.dual_timescale_low_rank_temporal import (
            load_model as load_254a_model,
        )

        return load_254a_model(checkpoint_path, device)
    if model_type == "254b":
        from diffusion.block_ar.anti_collapse_dual_timescale_temporal import (
            load_model as load_254b_model,
        )

        return load_254b_model(checkpoint_path, device)
    if model_type == "255a":
        from diffusion.block_ar.sparse_future_motif_model import (
            load_model as load_255a_model,
        )

        return load_255a_model(checkpoint_path, device)
    if model_type == "256a":
        from diffusion.block_ar.hierarchical_future_token_model import (
            load_model as load_256a_model,
        )

        return load_256a_model(checkpoint_path, device)
    if model_type in {"257a", "257b", "257c", "257d"}:
        from diffusion.block_ar.latent_future_token_vae import (
            load_model as load_257a_model,
        )

        return load_257a_model(checkpoint_path, device)
    if model_type == "258a":
        from diffusion.block_ar.stochastic_dual_timescale_ssm import (
            load_model as load_258a_model,
        )

        return load_258a_model(checkpoint_path, device)
    if model_type in {"260a", "260b", "260c", "260d", "260e", "260f"}:
        from diffusion.block_ar.minimal_factor_fm import (
            load_model as load_260a_model,
        )

        return load_260a_model(checkpoint_path, device)
    if model_type == "261a":
        from diffusion.block_ar.residual_scenario_fm import (
            load_model as load_261a_model,
        )

        return load_261a_model(checkpoint_path, device)
    if model_type in {"261b", "261c"}:
        from diffusion.block_ar.latent_factor_residual_fm import (
            load_model as load_261b_model,
        )

        return load_261b_model(checkpoint_path, device)
    if model_type == "261d":
        from diffusion.block_ar.scale_calibrated_latent_factor_residual_fm import (
            load_model as load_261d_model,
        )

        return load_261d_model(checkpoint_path, device)
    if model_type == "262a":
        from diffusion.block_ar.joint_probabilistic_latent_factor_fm import (
            load_model as load_262a_model,
        )

        return load_262a_model(checkpoint_path, device)
    if model_type == "262b":
        from diffusion.block_ar.joint_probabilistic_latent_factor_fm_ec import (
            load_model as load_262b_model,
        )

        return load_262b_model(checkpoint_path, device)
    if model_type == "263a":
        from diffusion.block_ar.joint_state_space_latent_factor_fm import (
            load_model as load_263a_model,
        )

        return load_263a_model(checkpoint_path, device)
    if model_type == "263b":
        from diffusion.block_ar.joint_state_space_latent_factor_fm_ec import (
            load_model as load_263b_model,
        )

        return load_263b_model(checkpoint_path, device)
    if model_type == "263c":
        from diffusion.block_ar.joint_state_space_latent_factor_fm_ec_scale_anchor import (
            load_model as load_263c_model,
        )

        return load_263c_model(checkpoint_path, device)
    if model_type == "264a":
        from diffusion.block_ar.joint_state_space_latent_factor_fm_posterior_teacher import (
            load_model as load_264a_model,
        )

        return load_264a_model(checkpoint_path, device)
    if model_type == "264b":
        from diffusion.block_ar.joint_state_space_latent_factor_fm_epsilon_teacher import (
            load_model as load_264b_model,
        )

        return load_264b_model(checkpoint_path, device)
    if model_type == "266a":
        from diffusion.block_ar.latent_bottleneck_diffusion import (
            load_model as load_266a_model,
        )

        return load_266a_model(checkpoint_path, device)
    if model_type == "266b":
        from diffusion.block_ar.latent_path_bottleneck_diffusion import (
            load_model as load_266b_model,
        )

        return load_266b_model(checkpoint_path, device)
    if model_type == "266c":
        from diffusion.block_ar.latent_path_bottleneck_diffusion_seq import (
            load_model as load_266c_model,
        )

        return load_266c_model(checkpoint_path, device)
    if model_type == "266d":
        from diffusion.block_ar.latent_path_bottleneck_flow_matching import (
            load_model as load_266d_model,
        )

        return load_266d_model(checkpoint_path, device)
    if model_type == "267a":
        from diffusion.block_ar.probabilistic_latent_token_model import (
            load_model as load_267a_model,
        )

        return load_267a_model(checkpoint_path, device)
    if model_type == "267b":
        from diffusion.block_ar.probabilistic_latent_token_model_reduced_bypass import (
            load_model as load_267b_model,
        )

        return load_267b_model(checkpoint_path, device)
    if model_type == "268a":
        from diffusion.block_ar.direct_path_flow_matching import (
            load_model as load_268a_model,
        )

        return load_268a_model(checkpoint_path, device)
    if model_type == "268b":
        from diffusion.block_ar.direct_path_flow_matching_xattn import (
            load_model as load_268b_model,
        )

        return load_268b_model(checkpoint_path, device)
    if model_type == "268c":
        from diffusion.block_ar.direct_change_flow_matching import (
            load_model as load_268c_model,
        )

        return load_268c_model(checkpoint_path, device)
    if model_type == "269a":
        from diffusion.block_ar.autoregressive_change_flow_matching import (
            load_model as load_269a_model,
        )

        return load_269a_model(checkpoint_path, device)
    if model_type == "270a":
        from diffusion.block_ar.ar_latent_bottleneck_flow_matching import (
            load_model as load_270a_model,
        )

        return load_270a_model(checkpoint_path, device)
    if model_type == "270b":
        from diffusion.block_ar.ar_latent_sequence_flow_matching import (
            load_model as load_270b_model,
        )

        return load_270b_model(checkpoint_path, device)
    if model_type == "270c":
        from diffusion.block_ar.ar_latent_sequence_change_flow_matching import (
            load_model as load_270c_model,
        )

        return load_270c_model(checkpoint_path, device)
    if model_type == "271a":
        from diffusion.block_ar.ar_latent_sequence_conditional_change_flow_matching import (
            load_model as load_271a_model,
        )

        return load_271a_model(checkpoint_path, device)
    if model_type == "272a":
        from diffusion.block_ar.ar_surface_latent_flow_matching import (
            load_model as load_272a_model,
        )

        return load_272a_model(checkpoint_path, device)
    if model_type == "272b":
        from diffusion.block_ar.ar_surface_latent_memory_flow_matching import (
            load_model as load_272b_model,
        )

        return load_272b_model(checkpoint_path, device)
    if model_type == "273a":
        from diffusion.block_ar.ar_surface_token_flow_matching import (
            load_model as load_273a_model,
        )

        return load_273a_model(checkpoint_path, device)
    if model_type == "274a":
        from diffusion.block_ar.ar_surface_probabilistic_token_model import (
            load_model as load_274a_model,
        )

        return load_274a_model(checkpoint_path, device)
    if model_type == "275a":
        from diffusion.block_ar.ar_deterministic_change_backbone import (
            load_model as load_275a_model,
        )

        return load_275a_model(checkpoint_path, device)
    if model_type == "275b":
        from diffusion.block_ar.ar_surface_deterministic_latent_backbone import (
            load_model as load_275b_model,
        )

        return load_275b_model(checkpoint_path, device)
    if model_type == "275c":
        from diffusion.block_ar.ar_seq2seq_transformer_change_backbone import (
            load_model as load_275c_model,
        )

        return load_275c_model(checkpoint_path, device)
    if model_type == "276a":
        from diffusion.block_ar.ar_seq2seq_transformer_change_token_backbone import (
            load_model as load_276a_model,
        )

        return load_276a_model(checkpoint_path, device)
    if model_type == "276b":
        from diffusion.block_ar.ar_seq2seq_transformer_joint_token_backbone import (
            load_model as load_276b_model,
        )

        return load_276b_model(checkpoint_path, device)
    if model_type == "277a":
        from diffusion.block_ar.deterministic_retrieval_backbone import (
            load_model as load_277a_model,
        )

        return load_277a_model(checkpoint_path, device)
    if model_type == "277b":
        from diffusion.block_ar.deterministic_retrieval_change_anchor_backbone import (
            load_model as load_277b_model,
        )

        return load_277b_model(checkpoint_path, device)
    if model_type == "277c":
        from diffusion.block_ar.deterministic_learned_retrieval_change_anchor_backbone import (
            load_model as load_277c_model,
        )

        return load_277c_model(checkpoint_path, device)
    if model_type == "277d":
        from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
            load_model as load_277d_model,
        )

        return load_277d_model(checkpoint_path, device)
    if model_type == "277e":
        from diffusion.block_ar.deterministic_learned_retrieval_medoid_backbone import (
            load_model as load_277e_model,
        )

        return load_277e_model(checkpoint_path, device)
    if model_type == "278a":
        from diffusion.block_ar.hierarchical_retrieval_scenario_generator import (
            load_model as load_278a_model,
        )

        return load_278a_model(checkpoint_path, device)
    if model_type == "278b":
        from diffusion.block_ar.hierarchical_retrieval_adaptive_scenario_generator import (
            load_model as load_278b_model,
        )

        return load_278b_model(checkpoint_path, device)
    if model_type == "278c":
        from diffusion.block_ar.hierarchical_retrieval_reweighted_scenario_generator import (
            load_model as load_278c_model,
        )

        return load_278c_model(checkpoint_path, device)
    if model_type == "279a":
        from diffusion.block_ar.hierarchical_retrieval_residual_scenario_generator import (
            load_model as load_279a_model,
        )

        return load_279a_model(checkpoint_path, device)
    if model_type == "279b":
        from diffusion.block_ar.hierarchical_retrieval_partial_residual_scenario_generator import (
            load_model as load_279b_model,
        )

        return load_279b_model(checkpoint_path, device)
    if model_type == "280a":
        from diffusion.block_ar.hierarchical_retrieval_scaled_residual_scenario_generator import (
            load_model as load_280a_model,
        )

        return load_280a_model(checkpoint_path, device)
    if model_type == "280b":
        from diffusion.block_ar.hierarchical_retrieval_horizon_scaled_residual_scenario_generator import (
            load_model as load_280b_model,
        )

        return load_280b_model(checkpoint_path, device)
    if model_type == "281a":
        from diffusion.block_ar.hierarchical_retrieval_temperature_residual_scenario_generator import (
            load_model as load_281a_model,
        )

        return load_281a_model(checkpoint_path, device)
    if model_type == "281b":
        from diffusion.block_ar.hierarchical_retrieval_temperature_scaled_residual_scenario_generator import (
            load_model as load_281b_model,
        )

        return load_281b_model(checkpoint_path, device)
    if model_type == "282a":
        from diffusion.block_ar.hierarchical_retrieval_temperature_scaled_residual_scenario_generator import (
            load_model as load_282a_model,
        )

        return load_282a_model(checkpoint_path, device)
    if model_type == "282b":
        from diffusion.block_ar.hierarchical_retrieval_temperature_scaled_residual_scenario_generator import (
            load_model as load_282b_model,
        )

        return load_282b_model(checkpoint_path, device)
    if model_type == "283a":
        from diffusion.block_ar.hierarchical_retrieval_reweighted_scenario_generator import (
            load_model as load_283a_model,
        )

        return load_283a_model(checkpoint_path, device)
    if model_type == "283b":
        from diffusion.block_ar.hierarchical_retrieval_reweighted_scenario_generator import (
            load_model as load_283b_model,
        )

        return load_283b_model(checkpoint_path, device)
    if model_type == "283c":
        from diffusion.block_ar.hierarchical_retrieval_reweighted_scenario_generator import (
            load_model as load_283c_model,
        )

        return load_283c_model(checkpoint_path, device)
    if model_type == "284b":
        from diffusion.block_ar.hierarchical_retrieval_reweighted_raw_future_scenario_generator import (
            load_model as load_284b_model,
        )

        return load_284b_model(checkpoint_path, device)
    if model_type == "285a":
        from diffusion.block_ar.hierarchical_retrieval_reweighted_offset_decay_scenario_generator import (
            load_model as load_285a_model,
        )

        return load_285a_model(checkpoint_path, device)
    if model_type == "286a":
        from diffusion.block_ar.hierarchical_retrieval_reweighted_history_affine_scenario_generator import (
            load_model as load_286a_model,
        )

        return load_286a_model(checkpoint_path, device)
    if model_type == "286b":
        from diffusion.block_ar.hierarchical_retrieval_reweighted_history_mean_scenario_generator import (
            load_model as load_286b_model,
        )

        return load_286b_model(checkpoint_path, device)
    if model_type == "287a":
        from diffusion.block_ar.deterministic_learned_retrieval_local_history_backbone import (
            load_model as load_287a_model,
        )

        return load_287a_model(checkpoint_path, device)
    if model_type == "287b":
        from diffusion.block_ar.deterministic_learned_retrieval_local_history_delta_backbone import (
            load_model as load_287b_model,
        )

        return load_287b_model(checkpoint_path, device)
    if model_type == "287c":
        from diffusion.block_ar.deterministic_learned_retrieval_local_history_two_timescale_delta_backbone import (
            load_model as load_287c_model,
        )

        return load_287c_model(checkpoint_path, device)
    if model_type == "287d":
        from diffusion.block_ar.deterministic_learned_retrieval_local_history_two_timescale_localz_delta_backbone import (
            load_model as load_287d_model,
        )

        return load_287d_model(checkpoint_path, device)
    if model_type == "287e":
        from diffusion.block_ar.deterministic_learned_retrieval_local_history_two_timescale_mixed_query_delta_backbone import (
            load_model as load_287e_model,
        )

        return load_287e_model(checkpoint_path, device)
    if model_type == "288a":
        from diffusion.block_ar.deterministic_soft_retrieval_local_history_mixed_query_delta_backbone import (
            load_model as load_288a_model,
        )

        return load_288a_model(checkpoint_path, device)
    if model_type == "289a":
        from diffusion.block_ar.deterministic_causal_transformer_world_model import (
            load_model as load_289a_model,
        )

        return load_289a_model(checkpoint_path, device)
    if model_type == "289b":
        from diffusion.block_ar.deterministic_latent_world_model import (
            load_model as load_289b_model,
        )

        return load_289b_model(checkpoint_path, device)
    if model_type == "289c":
        from diffusion.block_ar.deterministic_obs_encoded_latent_world_model import (
            load_model as load_289c_model,
        )

        return load_289c_model(checkpoint_path, device)
    if model_type == "289d":
        from diffusion.block_ar.deterministic_token_latent_world_model import (
            load_model as load_289d_model,
        )

        return load_289d_model(checkpoint_path, device)
    if model_type == "289e":
        from diffusion.block_ar.deterministic_history_memory_world_model import (
            load_model as load_289e_model,
        )

        return load_289e_model(checkpoint_path, device)
    if model_type == "290a":
        from diffusion.block_ar.deterministic_history_memory_discrete_world_model import (
            load_model as load_290a_model,
        )

        return load_290a_model(checkpoint_path, device)
    if model_type == "290b":
        from diffusion.block_ar.deterministic_history_memory_discrete_world_model import (
            load_model_soft_decode as load_290b_model,
        )

        return load_290b_model(checkpoint_path, device)
    if model_type == "291a":
        from diffusion.block_ar.deterministic_history_memory_joint_support_world_model import (
            load_model as load_291a_model,
        )

        return load_291a_model(checkpoint_path, device)
    if model_type == "293a":
        from diffusion.block_ar.probabilistic_joint_token_path_model import (
            load_model as load_293a_model,
        )

        return load_293a_model(checkpoint_path, device)
    if model_type == "293b":
        from diffusion.block_ar.probabilistic_joint_token_path_latent_model import (
            load_model as load_293b_model,
        )

        return load_293b_model(checkpoint_path, device)
    if model_type == "293c":
        from diffusion.block_ar.probabilistic_joint_token_path_knot_latent_model import (
            load_model as load_293c_model,
        )

        return load_293c_model(checkpoint_path, device)
    if model_type == "293d":
        from diffusion.block_ar.probabilistic_joint_token_path_support_model import (
            load_model as load_293d_model,
        )

        return load_293d_model(checkpoint_path, device)
    if model_type == "293e":
        from diffusion.block_ar.probabilistic_joint_token_path_knot_support_model import (
            load_model as load_293e_model,
        )

        return load_293e_model(checkpoint_path, device)
    if model_type == "293f":
        from diffusion.block_ar.probabilistic_joint_token_path_anchor_refine_model import (
            load_model as load_293f_model,
        )

        return load_293f_model(checkpoint_path, device)
    if model_type == "293g":
        from diffusion.block_ar.probabilistic_joint_token_path_support_residual_model import (
            load_model as load_293g_model,
        )

        return load_293g_model(checkpoint_path, device)
    if model_type == "293h":
        from diffusion.block_ar.probabilistic_joint_token_path_support_increment_residual_model import (
            load_model as load_293h_model,
        )

        return load_293h_model(checkpoint_path, device)
    if model_type == "294a":
        from diffusion.block_ar.probabilistic_joint_token_path_basis_residual_model import (
            load_model as load_294a_model,
        )

        return load_294a_model(checkpoint_path, device)
    if model_type == "294b":
        from diffusion.block_ar.probabilistic_joint_token_path_knot_residual_model import (
            load_model as load_294b_model,
        )

        return load_294b_model(checkpoint_path, device)
    if model_type == "295a":
        from diffusion.block_ar.probabilistic_joint_token_future_control_model import (
            load_model as load_295a_model,
        )

        return load_295a_model(checkpoint_path, device)
    if model_type == "296a":
        from diffusion.block_ar.probabilistic_backbone_residual_shell_model import (
            load_model as load_296a_model,
        )

        return load_296a_model(checkpoint_path, device)
    if model_type == "296b":
        from diffusion.block_ar.probabilistic_backbone_zero_mean_coarse_shell_model import (
            load_model as load_296b_model,
        )

        return load_296b_model(checkpoint_path, device)
    if model_type == "296c":
        from diffusion.block_ar.probabilistic_backbone_profiled_zero_mean_coarse_shell_model import (
            load_model as load_296c_model,
        )

        return load_296c_model(checkpoint_path, device)
    if model_type == "296e":
        from diffusion.block_ar.probabilistic_backbone_profiled_zero_mean_coarse_shell_model import (
            load_model as load_296e_model,
        )

        return load_296e_model(checkpoint_path, device)
    if model_type == "296f":
        from diffusion.block_ar.probabilistic_backbone_budgeted_multiresolution_zero_mean_coarse_shell_model import (
            load_model as load_296f_model,
        )

        return load_296f_model(checkpoint_path, device)
    if model_type == "296g":
        from diffusion.block_ar.probabilistic_backbone_gated_budgeted_multiresolution_zero_mean_coarse_shell_model import (
            load_model as load_296g_model,
        )

        return load_296g_model(checkpoint_path, device)
    if model_type == "298a":
        from diffusion.block_ar.probabilistic_structural_embedding_center_model import (
            load_model as load_298a_model,
        )

        return load_298a_model(checkpoint_path, device)
    if model_type == "299a":
        from diffusion.block_ar.local_scale_joint_change_flow_matching import (
            load_model as load_299a_model,
        )

        return load_299a_model(checkpoint_path, device)
    if model_type == "300a":
        from diffusion.block_ar.logit_level_flow_matching import (
            load_model as load_300a_model,
        )

        return load_300a_model(checkpoint_path, device)
    if model_type == "301a":
        from diffusion.block_ar.logit_transition_flow_matching import (
            load_model as load_301a_model,
        )

        return load_301a_model(checkpoint_path, device)
    if model_type == "302a":
        from diffusion.block_ar.logit_transition_state_flow_matching import (
            load_model as load_302a_model,
        )

        return load_302a_model(checkpoint_path, device)
    if model_type == "303a":
        from diffusion.block_ar.recurrent_logit_transition_flow_matching import (
            load_model as load_303a_model,
        )

        return load_303a_model(checkpoint_path, device)
    if model_type in {"303b", "329a", "329b"}:
        from diffusion.block_ar.recurrent_logit_transition_token_flow_matching import (
            load_model as load_303b_model,
        )

        return load_303b_model(checkpoint_path, device)
    if model_type in {"330a", "330b", "330c", "335a", "338a", "338b"}:
        from diffusion.block_ar.causal_future_memory_transition_flow_matching import (
            load_model as load_330a_model,
        )

        return load_330a_model(checkpoint_path, device)
    if model_type == "331a":
        from diffusion.block_ar.causal_memory_exact_transition_density import (
            load_model as load_331a_model,
        )

        return load_331a_model(checkpoint_path, device)
    if model_type == "332a":
        from diffusion.block_ar.causal_memory_scaled_source_transition_flow_matching import (
            load_model as load_332a_model,
        )

        return load_332a_model(checkpoint_path, device)
    if model_type in {"333a", "333b", "333c"}:
        from diffusion.block_ar.conditional_masked_path_flow_matching import (
            load_model as load_333a_model,
        )

        return load_333a_model(checkpoint_path, device)
    if model_type in {"339a", "339b"}:
        from diffusion.block_ar.empirical_normal_score_path_flow_matching import (
            load_model as load_339a_model,
        )

        return load_339a_model(checkpoint_path, device)
    if model_type == "421a":
        from diffusion.block_ar.empirical_normal_score_transition_path_flow_matching import (
            load_model as load_421a_model,
        )

        return load_421a_model(checkpoint_path, device)
    if model_type == "345a":
        from diffusion.block_ar.empirical_normal_score_masked_future_path_flow_matching import (
            load_model as load_345a_model,
        )

        return load_345a_model(checkpoint_path, device)
    if model_type == "346a":
        from diffusion.block_ar.empirical_normal_score_path_coupling_density import (
            load_model as load_346a_model,
        )

        return load_346a_model(checkpoint_path, device)
    if model_type in {"347a", "348a", "349a", "350a", "351a"}:
        from diffusion.block_ar.empirical_normal_score_transition_coupling_density import (
            load_model as load_347a_351a_model,
        )

        return load_347a_351a_model(checkpoint_path, device)
    if model_type in {"340a", "340b", "340c", "340d", "342a", "344a", "353a"}:
        from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
            load_model as load_340a_model,
        )

        return load_340a_model(checkpoint_path, device)
    if model_type == "544a":
        from diffusion.block_ar.local_shift_normalized_empirical_score_transition_flow_matching import (
            load_model as load_544a_model,
        )

        return load_544a_model(checkpoint_path, device)
    if model_type == "352a":
        from diffusion.block_ar.empirical_normal_score_causal_memory_transition_diffusion import (
            load_model as load_352a_model,
        )

        return load_352a_model(checkpoint_path, device)
    if model_type == "354a":
        from diffusion.block_ar.empirical_normal_score_path_diffusion import (
            load_model as load_354a_model,
        )

        return load_354a_model(checkpoint_path, device)
    if model_type == "343a":
        from diffusion.block_ar.empirical_normal_score_scalar_ar_mixture_density import (
            load_model as load_343a_model,
        )

        return load_343a_model(checkpoint_path, device)
    if model_type == "460a":
        from diffusion.block_ar.empirical_normal_score_day_vector_density import (
            load_model as load_460a_model,
        )

        return load_460a_model(checkpoint_path, device)
    if model_type == "470a":
        from diffusion.block_ar.frozen_center_residual_score_flow import (
            load_model as load_470a_model,
        )

        return load_470a_model(checkpoint_path, device)
    if model_type == "471a":
        from diffusion.block_ar.frozen_center_source_transport_score_flow import (
            load_model as load_471a_model,
        )

        return load_471a_model(checkpoint_path, device)
    if model_type == "476a":
        from diffusion.block_ar.latent_path_manifold_flow import (
            load_model as load_476a_model,
        )

        return load_476a_model(checkpoint_path, device)
    if model_type == "481a":
        from diffusion.block_ar.endpoint_conditioned_ar_bridge_flow import (
            load_model as load_481a_model,
        )

        return load_481a_model(checkpoint_path, device)
    if model_type == "483a":
        from diffusion.block_ar.frozen_proposal_density_ratio_resampler import (
            load_model as load_483a_model,
        )

        return load_483a_model(checkpoint_path, device)
    if model_type == "492a":
        from diffusion.block_ar.rank_copula_conditional_marginal_model import (
            load_model as load_492a_model,
        )

        return load_492a_model(checkpoint_path, device)
    if model_type == "341a":
        from diffusion.block_ar.empirical_normal_score_block_causal_flow_matching import (
            load_model as load_341a_model,
        )

        return load_341a_model(checkpoint_path, device)
    if model_type == "334a":
        from diffusion.block_ar.causal_memory_path_common_source_transition_flow_matching import (
            load_model as load_334a_model,
        )

        return load_334a_model(checkpoint_path, device)
    if model_type == "336a":
        from diffusion.block_ar.causal_memory_level_flow_matching import (
            load_model as load_336a_model,
        )

        return load_336a_model(checkpoint_path, device)
    if model_type == "337a":
        from diffusion.block_ar.causal_memory_exact_level_density import (
            load_model as load_337a_model,
        )

        return load_337a_model(checkpoint_path, device)
    if model_type == "303d":
        from diffusion.block_ar.recurrent_logit_transition_token_flow_matching import (
            load_model as load_303d_model,
        )

        return load_303d_model(checkpoint_path, device)
    if model_type == "303c":
        from diffusion.block_ar.recurrent_logit_transition_prefix_token_flow_matching import (
            load_model as load_303c_model,
        )

        return load_303c_model(checkpoint_path, device)
    if model_type == "304a":
        from diffusion.block_ar.joint_token_logit_transition_flow_matching import (
            load_model as load_304a_model,
        )

        return load_304a_model(checkpoint_path, device)
    if model_type == "304b":
        from diffusion.block_ar.joint_state_token_logit_transition_flow_matching import (
            load_model as load_304b_model,
        )

        return load_304b_model(checkpoint_path, device)
    if model_type == "305a":
        from diffusion.block_ar.latent_state_token_logit_transition_flow_matching import (
            load_model as load_305a_model,
        )

        return load_305a_model(checkpoint_path, device)
    if model_type == "306a":
        from diffusion.block_ar.recurrent_student_t_transition_likelihood import (
            load_model as load_306a_model,
        )

        return load_306a_model(checkpoint_path, device)
    if model_type == "307a":
        from diffusion.block_ar.recurrent_path_latent_token_likelihood import (
            load_model as load_307a_model,
        )

        return load_307a_model(checkpoint_path, device)
    if model_type == "308a":
        from diffusion.block_ar.probabilistic_obs_world_profiled_zero_mean_coarse_shell_model import (
            load_model as load_308a_model,
        )

        return load_308a_model(checkpoint_path, device)
    if model_type == "309a":
        from diffusion.block_ar.center_conditioned_residual_logit_transition_flow_matching import (
            load_model as load_309a_model,
        )

        return load_309a_model(checkpoint_path, device)
    if model_type == "310a":
        from diffusion.block_ar.center_conditioned_residual_logit_level_flow_matching import (
            load_model as load_310a_model,
        )

        return load_310a_model(checkpoint_path, device)
    if model_type == "311a":
        from diffusion.block_ar.center_conditioned_state_aware_residual_logit_level_flow_matching import (
            load_model as load_311a_model,
        )

        return load_311a_model(checkpoint_path, device)
    if model_type == "312a":
        from diffusion.block_ar.unified_state_aware_future_logit_path_flow_matching import (
            load_model as load_312a_model,
        )

        return load_312a_model(checkpoint_path, device)
    if model_type == "312b":
        from diffusion.block_ar.unified_history_memory_state_aware_future_logit_path_flow_matching import (
            load_model as load_312b_model,
        )

        return load_312b_model(checkpoint_path, device)
    if model_type == "312c":
        from diffusion.block_ar.unified_pooled_history_memory_state_aware_future_logit_path_flow_matching import (
            load_model as load_312c_model,
        )

        return load_312c_model(checkpoint_path, device)
    if model_type == "313a":
        from diffusion.block_ar.unified_latent_token_future_logit_path_flow_matching import (
            load_model as load_313a_model,
        )

        return load_313a_model(checkpoint_path, device)
    if model_type == "313b":
        from diffusion.block_ar.unified_probabilistic_latent_token_future_logit_path_model import (
            load_model as load_313b_model,
        )

        return load_313b_model(checkpoint_path, device)
    if model_type == "314a":
        from diffusion.block_ar.history_conditioned_probabilistic_token_state_logit_path_model import (
            load_model as load_314a_model,
        )

        return load_314a_model(checkpoint_path, device)
    if model_type == "314b":
        from diffusion.block_ar.history_conditioned_probabilistic_token_state_logit_transition_model import (
            load_model as load_314b_model,
        )

        return load_314b_model(checkpoint_path, device)
    if model_type == "314c":
        from diffusion.block_ar.history_conditioned_level_aware_probabilistic_token_state_logit_transition_model import (
            load_model as load_314c_model,
        )

        return load_314c_model(checkpoint_path, device)
    if model_type == "315a":
        from diffusion.block_ar.multiscale_future_logit_path_flow_matching import (
            load_model as load_315a_model,
        )

        return load_315a_model(checkpoint_path, device)
    if model_type == "315b":
        from diffusion.block_ar.multiscale_global_token_future_logit_path_flow_matching import (
            load_model as load_315b_model,
        )

        return load_315b_model(checkpoint_path, device)
    if model_type in {"316a", "324a"}:
        from diffusion.block_ar.unified_global_token_state_aware_future_logit_path_flow_matching import (
            load_model as load_316a_model,
        )

        return load_316a_model(checkpoint_path, device)
    if model_type == "324c":
        from diffusion.block_ar.axial_mixer_future_logit_path_flow_matching import (
            load_model as load_324c_model,
        )

        return load_324c_model(checkpoint_path, device)
    if model_type in {"325a", "325b"}:
        from diffusion.block_ar.energy_score_implicit_future_logit_path_generator import (
            load_model as load_325a_model,
        )

        return load_325a_model(checkpoint_path, device)
    if model_type in {"326a", "326b", "326c"}:
        from diffusion.block_ar.shared_latent_implicit_future_logit_path_generator import (
            load_model as load_326a_model,
        )

        return load_326a_model(checkpoint_path, device)
    if model_type in {"327a", "327b", "327c"}:
        from diffusion.block_ar.latent_bottleneck_future_logit_path_flow import (
            load_model as load_327a_model,
        )

        return load_327a_model(checkpoint_path, device)
    if model_type == "328a":
        from diffusion.block_ar.recurrent_latent_state_future_logit_path_flow import (
            load_model as load_328a_model,
        )

        return load_328a_model(checkpoint_path, device)
    if model_type in {"317a", "317b"}:
        from diffusion.block_ar.shared_stochastic_state_aware_future_logit_path_flow_matching import (
            load_model as load_317a_model,
        )

        return load_317a_model(checkpoint_path, device)
    if model_type in {"318a", "318b", "319a", "319b", "324b"}:
        from diffusion.block_ar.daily_joint_cholesky_transition_model import (
            load_model as load_318a_model,
        )

        return load_318a_model(checkpoint_path, device)
    if model_type in {
        "320a",
        "320b",
        "320c",
        "320d",
        "320e",
        "320f",
        "321a",
        "321b",
        "321c",
        "322a",
        "323a",
        "323b",
        "323c",
        "323d",
    }:
        if model_type in {"323a", "323b", "323c", "323d"}:
            from diffusion.block_ar.conditional_marginal_copula_model import (
                load_model as load_320a_model,
            )
        else:
            from diffusion.block_ar.future_scalar_ar_mixture_density_model import (
                load_model as load_320a_model,
            )

        return load_320a_model(checkpoint_path, device)
    if model_type == "296d":
        from diffusion.block_ar.probabilistic_backbone_student_t_zero_mean_coarse_shell_model import (
            load_model as load_296d_model,
        )

        return load_296d_model(checkpoint_path, device)
    if model_type.startswith("253"):
        from diffusion.block_ar.dynamic_change_factor_ssm import load_model as load_253a_model

        return load_253a_model(checkpoint_path, device)
    loader = _get_h1_loader(model_type)
    return loader(checkpoint_path, device)


def select_rollout_indices(
    *,
    test_start: int,
    val_size: int,
    history_len: int,
    future_len: int,
    max_windows: int | None,
    split: str,
) -> np.ndarray:
    max_train_idx = int(test_start) - int(history_len) - int(future_len)
    train_indices = np.arange(0, max_train_idx - int(val_size))
    val_indices = np.arange(max_train_idx - int(val_size), max_train_idx)
    if split == "train":
        indices = train_indices
        if max_windows is not None:
            indices = indices[: int(max_windows)]
    elif split == "train_tail":
        n = int(val_size) if max_windows is None else int(max_windows)
        indices = train_indices[-n:]
    elif split == "val":
        indices = val_indices
        if max_windows is not None:
            indices = indices[: int(max_windows)]
    else:
        raise ValueError(f"unknown split {split!r}")
    return indices.astype(np.int64)


def build_rollout_windows(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    max_windows: int | None,
    device: torch.device,
    split: str = "val",
) -> RolloutBatch:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    indices = select_rollout_indices(
        test_start=test_start,
        val_size=val_size,
        history_len=history_len,
        future_len=future_len,
        max_windows=max_windows,
        split=split,
    )

    history_01, future_01 = build_multistep_windows(indices, surf_tensor, history_len, future_len)
    history_norm = normalize_iv(history_01)
    future_norm = normalize_iv(future_01)
    future_01 = future_01.view(history_01.shape[0], future_len, 5, 5)
    return RolloutBatch(
        history_01=history_01,
        history_norm=history_norm,
        future_01=future_01,
        future_norm=future_norm.view(history_01.shape[0], future_len, 5, 5),
    )


def rollout_samples_in_batches(
    wrapper: OneDayKernelRolloutWrapper,
    history_norm: torch.Tensor,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, history_norm.shape[0], batch_size):
        end = min(start + batch_size, history_norm.shape[0])
        batch_hist = history_norm[start:end]
        samples = wrapper.sample_batched(
            batch_hist,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        )
        outputs.append(samples.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def evaluate_rollout_subset(
    wrapper: OneDayKernelRolloutWrapper,
    history_norm: torch.Tensor,
    future_01: torch.Tensor,
    batch_size: int,
    n_samples: int,
    n_steps: int,
    chunk_size: int,
    horizons: Iterable[int] = (1, 5, 10, 20, 30),
) -> dict[str, float]:
    samples = rollout_samples_in_batches(
        wrapper=wrapper,
        history_norm=history_norm,
        n_samples=n_samples,
        n_steps=n_steps,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )
    results: dict[str, float] = {}
    widths_h30 = []
    hist_01 = denormalize_iv(history_norm).cpu().numpy()
    mean_iv_hist = hist_01.mean(axis=(2, 3))
    vov = np.diff(mean_iv_hist, axis=1).std(axis=1)
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm = vov <= q20
    turb = vov >= q80

    for h in horizons:
        if h > future_01.shape[1]:
            continue
        gt_h = future_01[:, h - 1].cpu().numpy()
        samp_h = samples[:, :, h - 1]
        lo = np.quantile(samp_h, 0.05, axis=1)
        hi = np.quantile(samp_h, 0.95, axis=1)
        cov = ((gt_h >= lo) & (gt_h <= hi)).mean()
        width = (hi - lo).mean()
        results[f"cov90_h{h}"] = float(cov)
        results[f"width90_h{h}"] = float(width)
        floor_h = float((samp_h <= 0.001).mean())
        ceil_h = float((samp_h >= 0.99).mean())
        results[f"at_floor_h{h}"] = floor_h
        results[f"at_ceiling_h{h}"] = ceil_h
        window_width = (hi - lo).mean(axis=(1, 2))
        if calm.any() and turb.any():
            results[f"turb_calm_ratio_h{h}"] = float(
                window_width[turb].mean() / max(window_width[calm].mean(), 1e-8)
            )
        if h == 30 or (h == max([hh for hh in horizons if hh <= future_01.shape[1]], default=h)):
            widths_h30 = window_width

    if len(widths_h30):
        results["width90_terminal_mean"] = float(np.mean(widths_h30))
    return results


def rollout_energy_score_levels(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    n_samples: int,
    rollout_horizons: Iterable[int],
    chunk_size: int,
):
    from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
        energy_score,
    )

    wrapper = OneDayKernelRolloutWrapper(model)
    rollout = wrapper.sample_batched(
        normalize_iv(history_01),
        n_samples=n_samples,
        n_steps=future_01.shape[1],
        chunk_size=chunk_size,
        history_is_normalized=True,
    )
    rollout_flat = rollout.view(rollout.shape[0], rollout.shape[1], rollout.shape[2], 25)
    target_flat = future_01.view(future_01.shape[0], future_01.shape[1], 25)
    losses = []
    for h in rollout_horizons:
        if 1 <= h <= future_01.shape[1]:
            losses.append(energy_score(rollout_flat[:, :, h - 1], target_flat[:, h - 1]))
    if not losses:
        raise ValueError("No valid rollout_horizons for rollout_energy_score_levels")
    loss = torch.stack(losses).mean()
    metrics = {
        "rollout_energy": loss.detach(),
        "rollout_terminal_std": rollout_flat[:, :, -1].std(dim=1).mean().detach(),
    }
    return loss, metrics


def write_markdown_summary(path: str | Any, title: str, lines: list[str]) -> None:
    from pathlib import Path

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n")


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj
