from typing import Optional, Any, Sequence, List, Tuple, Dict
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
try:
    from adam_atan2_pytorch import AdamAtan2 # type: ignore
except ModuleNotFoundError:
    from torch.optim import AdamW as AdamATan2  # type: ignore

    print(
        "WARNING: adam_atan2_backend not available; falling back to torch.optim.AdamW."
        " Install/build adam-atan2 for best performance."
    )


from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    checkpoint_interval: int = 500  # save resume checkpoint every N optimizer steps
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    # ---- Q7 (trainable basin width). Pass on the command line with a "+" prefix (+q7_reg=jac ...).
    # q7_reg: none | jac | sens | noise | qat | clip | lsq. Active from step q7_start_frac*total_steps; its strength
    # ramps linearly from 0 to full over q7_ramp_steps (jac lambda, noise sigma, clip threshold).
    q7_reg: str = "none"
    q7_start_frac: float = 0.0
    q7_start_step: int = -1         # if >= 0, overrides q7_start_frac (fork arms start exactly at the fork step)
    q7_ramp_steps: int = 0
    q7_lambda: float = 0.0          # jac: weight of the Hutchinson estimate of ||J||_F^2 / dim of one L-level step
    q7_jac_sub: int = 128           # jac: examples of the batch used for the estimate
    q7_jac_every: int = 1           # jac: compute the penalty every k steps (scaled by k)
    q7_sigma: float = 0.0           # noise: Gaussian weight noise, std = sigma x per-tensor weight std (fresh each step)
                                    # sens: the weight-noise level sigma_p of the sensitivity probe
    q7_sens_sub: int = 64           # sens: examples of the batch used for the probe
    q7_bits: int = 4                # qat: per-tensor symmetric fake-quant bits (straight-through estimator)
    q7_clip_k: float = 0.0          # clip: |W| <= k x per-tensor std after every optimizer step
    q7_eval_batches: int = 3        # in-run eval on the first N test batches (0 = the full 422k test set, upstream behaviour)
    q7_fork_iters: List[int] = []   # after eval iteration i, keep resume_iter{i}.pt (a fork point for late-phase arms)
    q7_fork_from: Optional[str] = None  # fresh start only: copy this resume_iter file to resume.pt, then resume from it
    q7_max_steps: int = 0           # smoke tests: stop after this many optimizer steps (0 = no limit)
    q7_inline_eval: bool = True     # after each checkpoint save, run q7_eval.evaluate_checkpoint in this process
    q7_eval_n: int = 1000           # puzzles for the inline evaluation
    q7_act_bits: int = 0            # >0: static per-tensor symmetric activation fake-quant at every CastedLinear
                                    # input (STE), scale = absmax / Qp over the first q7_act_calib_steps steps,
                                    # then frozen and stored as buffer q7_act_s (runs uncompiled)
    q7_act_calib_steps: int = 20

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Q7: weight-LSQ (learned per-tensor step size, Esser et al. 2020) on every CastedLinear
        if config.q7_reg == "lsq":
            q7_install_lsq(model, config.q7_bits)
        if config.q7_act_bits > 0:
            q7_install_actq(model, config.q7_act_bits)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                q7_param_groups(model, config),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                q7_param_groups(model, config),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    path = os.path.join(config.checkpoint_path, f"step_{train_state.step}")
    if os.path.exists(path):
        # Q7: a restart replayed steps after this checkpoint was first written; its evaluation belongs to the
        # old file and is dropped so the new file is evaluated from scratch
        for stale in (os.path.join(config.checkpoint_path, f"eval_step_{train_state.step}.json"), path + ".nolsq"):
            if os.path.exists(stale):
                os.remove(stale)
                print(f"[q7] removed stale {stale}")
    torch.save(train_state.model.state_dict(), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        # Q7: a compiled run's checkpoint loaded into an uncompiled model (activation fake-quant runs)
        if not any(k.startswith("_orig_mod.") for k in model.state_dict()):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resizing puzzle embedding. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Create new tensor with expected shape, initialized to mean
                new_weights = torch.empty(expected_shape, dtype=puzzle_emb.dtype, device=puzzle_emb.device)
                mean_emb = torch.mean(puzzle_emb, dim=0)
                new_weights[:] = mean_emb

                # Copy all overlapping rows to preserve learned weights
                min_rows = min(puzzle_emb.shape[0], expected_shape[0])
                new_weights[:min_rows] = puzzle_emb[:min_rows]
                state_dict[puzzle_emb_name] = new_weights

        model.load_state_dict(state_dict, assign=True)


def save_resume_checkpoint(
    config: PretrainConfig,
    train_state: TrainState,
    ema_helper: Any,
    iter_id: int,
    batch_count: int,
    name: str = "resume.pt",
) -> None:
    """Atomically saves full resume state to {checkpoint_path}/resume.pt.

    Writes to a .tmp file first and renames atomically so a mid-write crash
    never leaves a corrupt checkpoint.  Call from rank 0 only.
    """
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    path = os.path.join(config.checkpoint_path, name)
    tmp_path = path + ".tmp"

    payload: dict = {
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_states": [opt.state_dict() for opt in train_state.optimizers],
        "step": train_state.step,
        "iter_id": iter_id,
        "batch_count": batch_count,
        # _iters equals iter_id at the start of the outer iteration; restore
        # this so the dataset RNG seed reproduces the same batch ordering.
        "dataset_iters": iter_id,
    }
    if ema_helper is not None:
        payload["ema_state"] = ema_helper.state_dict()

    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX, best-effort on Windows


def load_resume_checkpoint(
    config: PretrainConfig,
    train_state: TrainState,
    ema_helper: Any,
    rank: int,
    world_size: int,
) -> Tuple[int, int, int]:
    """Restores resume state on rank 0 then syncs model weights to all ranks.

    Returns ``(start_iter_id, skip_batches, dataset_iters)``.  Returns
    ``(0, 0, 0)`` when no resume checkpoint exists (fresh run).

    Optimizer states and EMA shadow weights are only restored on rank 0.
    Other ranks start with default optimizer state; because gradients are
    all-reduced before every optimizer step the momentums converge within
    a few steps.
    """
    resume_path = (
        os.path.join(config.checkpoint_path, "resume.pt")
        if config.checkpoint_path is not None
        else None
    )
    info = [0, 0, 0]  # [start_iter_id, skip_batches, dataset_iters]

    if rank == 0 and resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        payload = torch.load(resume_path, map_location="cuda")

        train_state.model.load_state_dict(payload["model_state_dict"], assign=True)
        for opt, sd in zip(train_state.optimizers, payload["optimizer_states"]):
            opt.load_state_dict(sd)
        train_state.step = payload["step"]

        if ema_helper is not None and "ema_state" in payload:
            ema_helper.load_state_dict(payload["ema_state"])

        info = [payload["iter_id"], payload["batch_count"], payload["dataset_iters"]]
        print(f"  Resumed: step={train_state.step}, outer_iter={info[0]}, skip_batches={info[1]}")

    if world_size > 1:
        # Sync model weights and buffers from rank 0 to all ranks
        with torch.no_grad():
            for p in list(train_state.model.parameters()) + list(train_state.model.buffers()):
                dist.broadcast(p, src=0)

        # Sync scalar resume info (step, iter_id, batch_count, dataset_iters)
        dist.broadcast_object_list(info, src=0)
        step_t = torch.tensor([train_state.step], dtype=torch.int64).cuda()
        dist.broadcast(step_t, src=0)
        train_state.step = int(step_t.item())

    return info[0], info[1], info[2]


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



# ----------------------------------------------------------------------------- Q7 regularizers
def q7_log(config: "PretrainConfig", rec: dict):
    """Append one JSON line to {checkpoint_path}/metrics.jsonl (the wandb run is offline and host-local)."""
    if config.checkpoint_path is None:
        return
    import json, time
    os.makedirs(config.checkpoint_path, exist_ok=True)
    rec = {"t": time.time(), **rec}
    with open(os.path.join(config.checkpoint_path, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _q7_linears(model: nn.Module):
    """Every CastedLinear weight (lm_head, q_head, token mixer mlp_t, channel mlp): the set the
    evaluation quantizes / perturbs (breadth_sweep.py `_linears`)."""
    return [m.weight for m in model.modules() if m.__class__.__name__ == "CastedLinear"]


def _q7_inner(model: nn.Module):
    """Eager (uncompiled) inner TRM module sharing the trained parameters."""
    m = getattr(model, "_orig_mod", model)
    return m.model.inner


def q7_inline_eval(config: "PretrainConfig", step: int):
    """Basin-width / INT4 evaluation of the checkpoint just saved (q7_eval.py, one level up). Errors are
    logged and never stop training."""
    import sys, traceback
    ck = os.path.join(config.checkpoint_path, f"step_{step}")
    out = os.path.join(config.checkpoint_path, f"eval_step_{step}.json")
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if here not in sys.path:
            sys.path.insert(0, here)
        import q7_eval
        rec = q7_eval.evaluate_checkpoint(ck, out, n=config.q7_eval_n, label=f"{config.run_name}/step_{step}")
        fp = next(r for r in rec["records"] if r["quant"]["spec"] == "fp32")
        w4t = next(r for r in rec["records"] if r["quant"]["spec"] == "w4t")
        q7_log(config, {"step": step, "kind": "q7_eval", "fp32": fp["pexact"], "w4t": w4t["pexact"],
                        "w4t_fidH": w4t["fidelity_H"], "jac_fro2": rec.get("jac_fro2"),
                        "jac_spec_mean": rec.get("jac_spec_mean")})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        q7_log(config, {"step": step, "kind": "q7_eval_error", "error": repr(e)})


def _q7_round_ste(x):
    return (x.round() - x).detach() + x


def _q7_grad_scale(x, g):
    return (x - x * g).detach() + x * g


def q7_install_lsq(model: nn.Module, bits: int):
    """Weight-side LSQ: CastedLinear.forward uses clamp(round(W/s), -Qp, Qp) * s with a learned scalar s
    per tensor (symmetric, Qp = 2^(b-1)-1, the same grid as the w4t evaluation), straight-through rounding,
    LSQ gradient scale 1/sqrt(N*Qp), init s = 2 mean|W| / sqrt(Qp). Modules without q7_lsq_s (e.g. the
    evaluation copies built by q7_eval) run the unmodified forward."""
    from models import layers
    qp = 2 ** (bits - 1) - 1

    def forward(self, input):
        w = self.weight
        s = getattr(self, "q7_lsq_s", None)
        if s is not None:
            ss = _q7_grad_scale(s, 1.0 / math.sqrt(w.numel() * qp))
            w = _q7_round_ste(torch.clamp(w / ss, -qp, qp)) * ss
        return torch.nn.functional.linear(input, w.to(input.dtype),
                                          bias=self.bias.to(input.dtype) if self.bias is not None else None)

    layers.CastedLinear.forward = forward
    for m in model.modules():
        if m.__class__.__name__ == "CastedLinear" and not hasattr(m, "q7_lsq_s"):
            W = m.weight.detach().float()
            m.q7_lsq_s = nn.Parameter((2 * W.abs().mean() / math.sqrt(qp)).reshape(()).clone())
    print(f"[q7] LSQ installed: {sum(1 for n, _ in model.named_parameters() if n.endswith('q7_lsq_s'))} step sizes")


def _q7_actq_hook(mod, args):
    x = args[0]
    if not getattr(mod, "_q7_frozen", False):          # calibration: observe absmax, pass through
        mod._q7_absmax = max(getattr(mod, "_q7_absmax", 0.0), float(x.detach().abs().max()))
        return None
    s = mod.q7_act_s.to(x.dtype)
    qp = mod._q7_act_qp
    return (_q7_round_ste(torch.clamp(x / s, -qp, qp)) * s,) + tuple(args[1:])


def q7_install_actq(model: nn.Module, bits: int):
    """Static per-tensor activation fake-quant at every CastedLinear input (the w+a setting of
    Ingolfsson et al.). Needs eager mode (forward pre-hooks); launch() sets DISABLE_COMPILE."""
    n = 0
    for m in model.modules():
        if m.__class__.__name__ == "CastedLinear":
            m.register_buffer("q7_act_s", torch.zeros((), device=m.weight.device), persistent=True)
            m._q7_act_qp = 2 ** (bits - 1) - 1
            m._q7_frozen = False
            m._q7_absmax = 0.0
            m.register_forward_pre_hook(_q7_actq_hook)
            n += 1
    print(f"[q7] activation fake-quant ({bits}-bit, per-tensor static) on {n} linears")


@torch.no_grad()
def q7_actq_freeze(model: nn.Module, from_buffers: bool = False):
    """End of calibration (or after a resume): fix every scale; from_buffers keeps loaded scales."""
    out = []
    for m in model.modules():
        if hasattr(m, "q7_act_s"):
            if not from_buffers:
                m.q7_act_s.fill_(max(m._q7_absmax, 1e-8) / m._q7_act_qp)
            m._q7_frozen = bool(m.q7_act_s.item() > 0)
            out.append(float(m.q7_act_s))
    return out


def q7_param_groups(model: nn.Module, config: "PretrainConfig"):
    """model.parameters() unless LSQ is on; then the step sizes form a group without weight decay."""
    if config.q7_reg != "lsq":
        return model.parameters()
    steps = [p for n, p in model.named_parameters() if n.endswith("q7_lsq_s")]
    rest = [p for n, p in model.named_parameters() if not n.endswith("q7_lsq_s")]
    return [{"params": rest}, {"params": steps, "weight_decay": 0.0}]


def q7_strength(config: PretrainConfig, train_state: TrainState) -> float:
    """0 before the regularizer starts, then a linear ramp to 1 over q7_ramp_steps."""
    if config.q7_reg == "none":
        return 0.0
    start = config.q7_start_step if config.q7_start_step >= 0 else int(config.q7_start_frac * train_state.total_steps)
    if train_state.step < start:
        return 0.0
    if config.q7_ramp_steps <= 0:
        return 1.0
    return min(1.0, (train_state.step - start + 1) / config.q7_ramp_steps)


@torch.no_grad()
def q7_perturb_(config: PretrainConfig, weights, strength: float):
    """Before the forward pass: noise -> W + eps; qat -> Q(W). Returns what restores W afterwards."""
    saved = []
    for W in weights:
        if config.q7_reg == "noise":
            eps = torch.randn_like(W) * (strength * config.q7_sigma * W.std())
            W.add_(eps)
            saved.append(eps)
        elif config.q7_reg == "qat":
            qmax = 2 ** (config.q7_bits - 1) - 1
            s = (W.abs().max() / qmax).clamp_min(1e-8)
            saved.append(W.detach().clone())
            # homotopy W + a (Q(W) - W), a = strength: a ramp (q7_ramp_steps > 0) introduces the quantizer
            # gradually during training (in-training compression, CompreSSM's lesson); a = 1 is plain QAT
            W.copy_(W + strength * (torch.round(W / s).clamp(-qmax, qmax) * s - W))
    return saved


@torch.no_grad()
def q7_restore_(config: PretrainConfig, weights, saved):
    for W, s in zip(weights, saved):
        if config.q7_reg == "noise":
            W.sub_(s)
        elif config.q7_reg == "qat":
            W.copy_(s)


@torch.no_grad()
def q7_clip_(config: PretrainConfig, weights, strength: float):
    """After the optimizer step: clamp |W| to k_t x std(W), k_t ramping from 20 (a no-op) to q7_clip_k."""
    k = config.q7_clip_k + (1.0 - strength) * (20.0 - config.q7_clip_k)
    for W in weights:
        t = float(k * W.std())
        W.clamp_(-t, t)


def q7_input_embeddings(inner: nn.Module, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor):
    """TRM's _input_embeddings without the sparse-embedding training path (which would overwrite
    the local buffers the puzzle-embedding optimizer reads this step)."""
    emb = inner.embed_tokens(inputs.to(torch.int32))
    if inner.config.puzzle_emb_ndim > 0:
        pe = inner.puzzle_emb.weights[puzzle_identifiers].to(emb.dtype)
        pad = inner.puzzle_emb_len * inner.config.hidden_size - pe.shape[-1]
        if pad > 0:
            pe = torch.nn.functional.pad(pe, (0, pad))
        emb = torch.cat((pe.view(-1, inner.puzzle_emb_len, inner.config.hidden_size), emb), dim=-2)
    if inner.config.pos_encodings == "learned":
        emb = 0.707106781 * (emb + inner.embed_pos.embedding_weight.to(emb.dtype))
    return inner.embed_scale * emb


def q7_jacobian_penalty(config: PretrainConfig, train_state: TrainState):
    """Hutchinson estimate of ||J||_F^2 / dim for one L-level recursion step z_L -> f(z_L, z_H + x),
    evaluated at the carry the batch just produced (Bai et al. 2021's DEQ Jacobian regulariser, applied
    to TRM's shared block, i.e. token mixer + channel MLP). Eager double backward; weights only."""
    inner = _q7_inner(train_state.model)
    c = train_state.carry
    n = min(config.q7_jac_sub, c.inner_carry.z_L.shape[0])
    with torch.no_grad():
        cd = c.current_data
        x = q7_input_embeddings(inner, cd["inputs"][:n], cd["puzzle_identifiers"][:n])
        inj = (c.inner_carry.z_H[:n] + x).detach()
    z = c.inner_carry.z_L[:n].detach().clone().requires_grad_(True)
    cos_sin = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None
    out = inner.L_level(z, inj, cos_sin=cos_sin)
    v = torch.randn_like(out)
    (g,) = torch.autograd.grad(out, z, v, create_graph=True)
    return g.float().pow(2).sum() / v.numel()


def q7_act_step(level: nn.Module, cfg, zH: torch.Tensor, zL: torch.Tensor, x: torch.Tensor, cs=None):
    """One full ACT step F of TRM (H_cycles x (L_cycles L-updates + 1 H-update)), with gradient throughout:
    the map whose one-step perturbation delta_T X2 measures (contraction.py, TRM family)."""
    for _ in range(cfg.H_cycles):
        for _ in range(cfg.L_cycles):
            zL = level(zL, zH + x, cos_sin=cs)
        zH = level(zH, zL, cos_sin=cs)
    return zH, zL


_Q7_SENS_COPY = {}


def q7_sensitivity_penalty(config: PretrainConfig, train_state: TrainState):
    """S = delta / sigma_p, delta = ||F_{W+eps}(s) - F_W(s)|| / ||s|| per puzzle (mean over puzzles), with
    eps = sigma_p x per-tensor std x N(0, 1) on every weight of the recursion block (fresh draw each step)
    and s the carry this batch just produced (the training-time end state). Differentiable in W through both
    maps: the perturbed map runs on a structural copy of L_level whose weights are the tensors W + eps
    (the trained module is never mutated, so torch.compile's guards are untouched). Eager, fp32.
    X2: S at fp32's end state explains 84% of the variance of log basin width across six models."""
    import copy as _copy
    inner = _q7_inner(train_state.model)
    level = inner.L_level
    if "copy" not in _Q7_SENS_COPY:
        cp = _copy.deepcopy(level)
        for m in cp.modules():
            if m.__class__.__name__ == "CastedLinear":
                del m._parameters["weight"]
        _Q7_SENS_COPY["copy"] = cp
    cp = _Q7_SENS_COPY["copy"]
    src = {nm: m for nm, m in level.named_modules() if m.__class__.__name__ == "CastedLinear"}
    for nm, m in cp.named_modules():
        if m.__class__.__name__ == "CastedLinear":
            W = src[nm].weight
            m.weight = W + torch.randn_like(W) * (config.q7_sigma * W.detach().std())
    c = train_state.carry
    n = min(config.q7_sens_sub, c.inner_carry.z_L.shape[0])
    with torch.no_grad():
        cd = c.current_data
        x = q7_input_embeddings(inner, cd["inputs"][:n], cd["puzzle_identifiers"][:n]).float()
        zH = c.inner_carry.z_H[:n].float()
        zL = c.inner_carry.z_L[:n].float()
    cs = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None
    h0, l0 = q7_act_step(level, inner.config, zH, zL, x, cs)
    h1, l1 = q7_act_step(cp, inner.config, zH, zL, x, cs)
    for m in cp.modules():                      # drop the graph references held by the copy
        if m.__class__.__name__ == "CastedLinear":
            m.weight = None
    num = torch.sqrt((h1 - h0).pow(2).flatten(1).sum(1) + (l1 - l0).pow(2).flatten(1).sum(1))
    den = torch.sqrt(zH.pow(2).flatten(1).sum(1) + zL.pow(2).flatten(1).sum(1)).clamp_min(1e-6)
    return (num / den).mean() / config.q7_sigma


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Q7: perturb the weights for this step's forward/backward (noise, qat)
    q7_s = q7_strength(config, train_state)
    q7_w = _q7_linears(train_state.model) if q7_s > 0 else []
    q7_saved = q7_perturb_(config, q7_w, q7_s) if config.q7_reg in ("noise", "qat") and q7_s > 0 else []

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    if q7_saved:
        q7_restore_(config, q7_w, q7_saved)   # gradient taken at the perturbed point, applied to the clean weights

    q7_jac = None
    if config.q7_reg == "jac" and q7_s > 0 and train_state.step % config.q7_jac_every == 0:
        q7_jac = q7_jacobian_penalty(config, train_state)
        (q7_s * config.q7_lambda * config.q7_jac_every * q7_jac).backward()
    if config.q7_reg == "sens" and q7_s > 0 and train_state.step % config.q7_jac_every == 0:
        q7_jac = q7_sensitivity_penalty(config, train_state)       # logged under q7/jac_fro2 = S estimate
        (q7_s * config.q7_lambda * config.q7_jac_every * q7_jac).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    if config.q7_reg == "clip" and q7_s > 0:
        q7_clip_(config, q7_w, q7_s)

    if config.q7_act_bits > 0 and not getattr(train_state, "_q7_act_frozen", False):
        if train_state.step >= getattr(train_state, "_q7_act_start", 0) + config.q7_act_calib_steps:
            sc = q7_actq_freeze(train_state.model)
            train_state._q7_act_frozen = True
            print(f"[q7] activation scales frozen at step {train_state.step}: {[round(x, 4) for x in sc]}")

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            reduced_metrics["q7/strength"] = q7_s
            if q7_jac is not None:
                reduced_metrics["q7/jac_fro2"] = float(q7_jac.detach())
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            if config.q7_eval_batches > 0 and processed_batches >= config.q7_eval_batches:
                break
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)
    if config.q7_act_bits > 0:
        os.environ["DISABLE_COMPILE"] = "1"   # activation fake-quant uses forward pre-hooks

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # EMA setup before resume load so shadow weights are registered before being restored
    ema_helper = None
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Q7: a late-phase arm starts from a fork point of its base run (resume_iter{i}.pt)
    if RANK == 0 and config.q7_fork_from and config.checkpoint_path is not None:
        rp = os.path.join(config.checkpoint_path, "resume.pt")
        if not os.path.exists(rp):
            os.makedirs(config.checkpoint_path, exist_ok=True)
            shutil.copy(config.q7_fork_from, rp + ".tmp")
            os.replace(rp + ".tmp", rp)
            print(f"[q7] forked from {config.q7_fork_from}")

    # Resume checkpoint — restores model weights, optimizer states, EMA, step, and
    # iter position.  Returns (0, 0, 0) on a fresh run.
    start_iter_id, skip_batches, dataset_iters = load_resume_checkpoint(
        config, train_state, ema_helper, rank=RANK, world_size=WORLD_SIZE
    )
    # Restore the dataset RNG counter so the resumed iteration uses the same
    # batch ordering as the original run.  Must be set before the DataLoader
    # worker is first spawned (i.e. before the training loop begins).
    train_loader.dataset._iters = dataset_iters
    if config.q7_act_bits > 0:
        sc = q7_actq_freeze(train_state.model, from_buffers=True)
        train_state._q7_act_start = train_state.step
        train_state._q7_act_frozen = all(x > 0 for x in sc)
        print(f"[q7] activation scales after load: {'frozen' if train_state._q7_act_frozen else 'calibrating'}")

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step, mininterval=60)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=train_state.step)
        save_code_and_config(config)

    # Q7: finish evaluations of checkpoints written before a restart (resumable per cell)
    if RANK == 0 and config.q7_inline_eval and config.checkpoint_path and os.path.isdir(config.checkpoint_path):
        import re as _re
        for fn in sorted(os.listdir(config.checkpoint_path)):
            mm = _re.fullmatch(r"step_(\d+)", fn)
            if mm and int(mm.group(1)) <= train_state.step:
                q7_inline_eval(config, int(mm.group(1)))

    # Training Loop
    for _iter_id in range(start_iter_id, total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        batch_count = 0
        for set_name, batch, global_batch_size in train_loader:
            # Fast-forward through already-processed batches in the resumed iteration
            if _iter_id == start_iter_id and batch_count < skip_batches:
                batch_count += 1
                continue

            batch_count += 1
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                if train_state.step % 50 == 0 or train_state.step <= 10:
                    q7_log(config, {"step": train_state.step, "kind": "train", **{k: float(v) for k, v in metrics.items()}})
            if config.ema:
                ema_helper.update(train_state.model)

            # Periodic resume checkpoint — atomic write, rank 0 only
            if RANK == 0 and train_state.step % config.checkpoint_interval == 0:
                save_resume_checkpoint(config, train_state, ema_helper, _iter_id, batch_count)

            if config.q7_max_steps and train_state.step >= config.q7_max_steps:
                break

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                q7_log(config, {"step": train_state.step, "kind": "eval", "eval_batches": config.q7_eval_batches,
                                **{f"{sn}/{k}": float(v) for sn, m in metrics.items() if isinstance(m, dict) for k, v in m.items()}})

            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)
                # Q7: resume point right after the checkpoint, so a restart during the (long) evaluation below
                # never replays steps and rewrites step_N under a finished or half-finished evaluation
                save_resume_checkpoint(config, train_state, ema_helper, _iter_id + 1, 0)
                if config.q7_inline_eval:
                    q7_inline_eval(config, train_state.step)
            if RANK == 0 and _iter_id in config.q7_fork_iters:
                # fork point: the state at the start of iteration _iter_id + 1 (no batches consumed)
                save_resume_checkpoint(config, train_state, ema_helper, _iter_id + 1, 0, name=f"resume_iter{_iter_id}.pt")
                print(f"[q7] fork point resume_iter{_iter_id}.pt at step {train_state.step}")

            if config.ema:
                del train_state_eval

        if config.q7_max_steps and train_state.step >= config.q7_max_steps:
            print(f"[q7] q7_max_steps={config.q7_max_steps} reached")
            break

    if RANK == 0 and config.checkpoint_path is not None:
        open(os.path.join(config.checkpoint_path, "DONE"), "w").write(
            f"{train_state.step}{' (q7_max_steps)' if config.q7_max_steps else ''}\n")

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
