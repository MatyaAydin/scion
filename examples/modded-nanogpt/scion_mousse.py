import math
import torch
import os

#######################################################
# Norm classes (unchanged from ScionShampoo)
#######################################################


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError


class ColNorm(Norm):
    def __init__(self, normalized=False, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1)
        rms_values = 1/math.sqrt(g.size(0))*torch.sqrt(torch.sum(g ** 2, dim=0, keepdim=True))
        if self.normalized:
            rms_values *= g.size(1)
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1)
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=0, keepdim=True)
        w.data *= math.sqrt(w.size(0))
        if self.normalized:
            w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return w


class RowNorm(Norm):
    def __init__(self, normalized=True, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1)
        rms_values = torch.sqrt(torch.sum(g ** 2, dim=-1, keepdim=True))
        if self.normalized:
            rms_values *= math.sqrt(g.size(-1))
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1)
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=-1, keepdim=True)
        if self.normalized:
            w.data /= math.sqrt(w.size(-1))
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return w


class BiasRMS(Norm):
    def lmo(self, g):
        eps = 1e-8
        rms_values = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        g = g / (rms_values + eps)
        return g

    def init(self, g):
        return torch.nn.init.zeros_(g)


class SpectralConv(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        if g.ndim == 3:
            out_channels, in_channels, k = g.shape
            g *= (out_channels / in_channels)**0.5 / k
        elif g.ndim == 4:
            out_channels, in_channels, k, _ = g.shape
            g *= (out_channels / in_channels)**0.5 / (k ** 2)
        return g

    def init(self, w):
        w_fp = w.data.double()
        k = w.data.size(2)
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:,:,kx,ky])
        if w.ndim == 3:
            out_channels, in_channels, k = w_fp.shape
            w_fp.mul_((out_channels / in_channels)**0.5 / k)
        elif w.ndim == 4:
            out_channels, in_channels, k, _ = w_fp.shape
            w_fp.mul_((out_channels / in_channels)**0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape
        scale = (d_out / d_in)**0.5 if self.normalized else d_out**0.5
        if self.max:
            scale = max(1, scale)
        g *= scale
        return g

    def init(self, w):
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp)
        d_out, d_in = w_fp.shape
        scale = (d_out / d_in)**0.5 if self.normalized else d_out**0.5
        if self.max:
            scale = max(1, scale)
        w_fp.mul_(scale)
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        d_out, d_in = g.shape
        if self.normalized:
            return (1/d_in)*torch.sign(g)
        else:
            return torch.sign(g)

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            d_out, d_in = w.shape
            w.data = (torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1)
            if self.normalized:
                w.data *= (1/d_in)
        return w


class Auto(Norm):
    def lmo(self, g):
        if g.ndim in [3, 4]:
            return SpectralConv().lmo(g)
        elif g.ndim == 2:
            return Spectral().lmo(g)
        elif g.ndim in [0, 1]:
            return BiasRMS().lmo(g)

    def init(self, w):
        if w.ndim in [3, 4]:
            return SpectralConv().init(w)
        elif w.ndim == 2:
            return Spectral().init(w)
        elif w.ndim in [0, 1]:
            return BiasRMS().init(w)


norm_dict = {
    'ColNorm': ColNorm,
    'RowNorm': RowNorm,
    'BiasRMS': BiasRMS,
    'SpectralConv': SpectralConv,
    'Spectral': Spectral,
    'Sign': Sign,
    'Auto': Auto,
}


#######################################################
# Shared helpers
#######################################################


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.float()


def to_2d(g):
    if g.dim() == 1:
        return g.unsqueeze(0)
    return g.reshape(g.shape[0], -1)


def clean_eigenvalues(evals, epsilon):
    min_eig = evals.min()
    shift = torch.clamp(-min_eig, min=0.0) + epsilon
    return evals + shift


#######################################################
# eig_update_freq scheduler
#######################################################


def get_eig_update_freq(t, eig_schedule):
    """
    Compute the effective eig_update_freq at step t given a schedule dict.

    Three-phase schedule:
      Phase 1  [0, eig_warmup_steps):
          Returns None  →  don't use preconditioning.

      Phase 2  [eig_warmup_steps, warmdown_start]:
          Returns T_init  →  frequent refreshes (e.g. 125).

      Phase 3  (warmdown_start, total_steps]:
          Returns None  →  don't use preconditioning during warmdown.

    Args:
        t (int):             Current global training step.
        eig_schedule (dict): Must contain:
            'eig_warmup_steps' (int)  — end of Phase 1.
            'warmdown_start'   (int)  — end of Phase 2 / start of Phase 3.
            'T_init'           (int)  — freq during Phase 2.

    Returns:
        int | None: Effective eig_update_freq for step t, or None to skip.
    """
    eig_warmup     = eig_schedule.get('eig_warmup_steps', 500)
    warmdown_start = eig_schedule.get('warmdown_start', 5250)
    T_init         = eig_schedule.get('T_init', 125)

    if t < eig_warmup:
        return None

    if t <= warmdown_start:
        return T_init

    return None


#######################################################
# MousseScion
#######################################################


class MousseScion(torch.optim.Optimizer):
    """
    Mousse-style L,R preconditioning applied to the Scion optimizer.

    Update equations (per step t):
        m_t  =  α_t · m_{t-1}  +  (1-α_t) · G_t                  [momentum]
        L_t  =  β · L_{t-1}   +  (1-β) · G_t G_t^T               [left  curvature EMA]
        R_t  =  β · R_{t-1}   +  (1-β) · G_t^T G_t               [right curvature EMA]
        (Λ_L, Q_L) = eigh(L̂_t),  (Λ_R, Q_R) = eigh(R̂_t)       [every effective T steps]
        M̃      =  L^{-p} m_t R^{-p}                               [preconditioned momentum]
        u      =  L^{-p} lmo(M̃) R^{-p}                            [preconditioned direction]
        w_{t+1} = (1-η) w_t − η s u                               [Frank-Wolfe update]

    Preconditioning strategy per layer:

    Spectral / SpectralConv layers — orthogonality trick
        lmo = msign (Newton-Schulz).  Since msign is applied between two orthogonal
        Q factors, the Q^T Q = I identity cancels:
            L^{-p} msign(M̃) R^{-p}  =  Q_L Λ_L^{-p} msign(Λ_L^{-p} Q_L^T M Q_R Λ_R^{-p}) Λ_R^{-p} Q_R^T
        The entire whitening / LMO / unwhitening lives in the compact eigenbasis;
        no full-size matrix is ever materialised.

    All other 2-D layers (Sign, ColNorm, RowNorm, …) — full rebuild
        lmo is not invariant to the pre/post multiplication by Q, so M̃ must be
        reconstructed explicitly in the original parameter space:
            M_tilde = Q_L Λ_L^{-p} Q_L^T  M  Q_R Λ_R^{-p} Q_R^T
            u_lmo   = lmo(M_tilde)
            O_k     = Q_L Λ_L^{-p} Q_L^T u_lmo Q_R Λ_R^{-p} Q_R^T

    One-sided fallback (large-vocab / memory-constrained layers)
        When one side exceeds max_precond_size the corresponding Gram matrix
        (L or R) is not formed.  Only the affordable side is used:
            M_tilde = L^{-p} M   (if only L fits)   or   M R^{-p}  (if only R fits)
        Spectral layers are always small enough that both sides fit.

    No preconditioning (both sides exceed max_precond_size)
        u = lmo(m_t)   — plain Scion.

    Args:
        params:                   Parameters to optimize.
        lr (float):               Learning rate η (default: 0.00036).
        momentum (float):         Momentum EMA coefficient (default: 0.9).
        norm (str):               LMO norm class (default: 'Auto').
        norm_kwargs (dict):       Extra kwargs for the norm class (default: {}).
        scale (float):            Constraint radius s (default: 1.0).
        unconstrained (bool):     Skip (1-lr) shrinkage (default: False).
        beta (float):             Curvature EMA decay (default: 0.99).
        alpha (float):            Curvature exponent p (default: 0.125).
        eps (float):              Eigenvalue damping (default: 1e-8).
        eig_update_freq (int):    Fixed eigh frequency. Used only when
                                  eig_schedule is None (default: 125).
        eig_schedule (dict|None): Frequency schedule. When set,
                                  eig_update_freq is ignored. See
                                  get_eig_update_freq() for full key docs.
                                  Example for a 7500-step run:
                                    {
                                      'eig_warmup_steps': 500,
                                      'warmdown_start':   5250,
                                      'T_init':           125,
                                    }
        use_trace_normalization (bool): Trace-normalise L,R before eigh (default: True).
        LR_correction (bool):     Bias-correct curvature EMAs (default: True).
        apply_grafting (str):     'fro', 'dual', 'ratio', 'interpolate', or 'lmo'
                                  (default: 'ratio').
        max_precond_size (int):   Maximum dimension for which a Gram matrix
                                  (L or R) is materialised.  Dimensions exceeding
                                  this threshold fall back to one-sided or no
                                  preconditioning (default: 8192).
    """

    def __init__(
        self,
        params,
        lr: float = 0.00036,
        momentum: float = 0.9,
        norm: str = 'Auto',
        norm_kwargs: dict = None,
        scale: float = 1.0,
        unconstrained: bool = False,
        beta: float = 0.99,
        alpha: float = 0.125,
        eps: float = 1e-8,
        eig_update_freq: int = 125,
        eig_schedule: dict | None = None,
        use_trace_normalization: bool = True,
        LR_correction: bool = True,
        apply_grafting: str = "ratio",
        norm_warmup_steps: int = 500,
        beta_scale: float = 0.9,
        max_precond_size: int = 8192,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0,1], got {momentum}.")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0,1), got {beta}.")
        if norm not in norm_dict:
            raise ValueError(f"Unknown norm '{norm}'. Choose from {list(norm_dict.keys())}.")
        if apply_grafting not in ("fro", "dual", "interpolate", "ratio", "lmo"):
            raise ValueError(f"apply_grafting must be 'fro' or 'dual', got '{apply_grafting}'.")
        if norm_kwargs is None:
            norm_kwargs = {}
        if eig_schedule is not None:
            for key in ('eig_warmup_steps', 'warmdown_start', 'T_init'):
                if key not in eig_schedule:
                    raise ValueError(f"eig_schedule is missing required key '{key}'.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            scale=scale,
            unconstrained=unconstrained,
            norm=norm,
            norm_kwargs=norm_kwargs,
            beta=beta,
            alpha=alpha,
            eps=eps,
            eig_update_freq=eig_update_freq,
            eig_schedule=eig_schedule,
            use_trace_normalization=use_trace_normalization,
            LR_correction=LR_correction,
            apply_grafting=apply_grafting,
            norm_warmup_steps=norm_warmup_steps,
            beta_scale=beta_scale,
            max_precond_size=max_precond_size,
        )
        super().__init__(params, defaults)
        self.effective_lrs = {}
        self.fro_norms = {}
        self.dual_norms = {}
        self.denom_norms = {}
        self.norm_ratios = {}

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr               = group['lr']
            momentum         = group['momentum']
            scale            = group['scale']
            unconstrained    = group['unconstrained']
            norm_backend     = norm_dict[group['norm']](**group['norm_kwargs'])
            beta             = group['beta']
            alpha            = group['alpha']
            eps              = group['eps']
            eig_update_freq  = group['eig_update_freq']
            eig_schedule     = group['eig_schedule']
            use_trace_norm   = group['use_trace_normalization']
            LR_correction    = group['LR_correction']
            apply_grafting   = group['apply_grafting']
            beta_scale       = group['beta_scale']
            max_precond_size = group['max_precond_size']
            # Spectral norms allow the Q^TQ=I orthogonality trick: lmo is applied
            # in the eigenbasis (whitened space) without needing to reconstruct the
            # full preconditioned matrix.  All other norms require a full rebuild
            # of M_tilde = L^{-p} M R^{-p} in the original space before lmo.
            is_spectral = group['norm'] in ('Spectral', 'SpectralConv')

            for p in group['params']:
                if p.grad is None:
                    continue

                g    = p.grad
                g_2d = to_2d(g).float()
                m, n = g_2d.shape
                state = self.state[p]

                # ── Which sides of the preconditioner can we afford? ──────────
                # L is m×m, R is n×n.  For a large-vocab projection the output
                # dimension can be ~50k, making one side unaffordable.
                # Spectral layers are small enough that both always fit; the check
                # is only relevant for non-Spectral (e.g. Sign, ColNorm …) layers.
                use_L = (m <= max_precond_size)
                use_R = (n <= max_precond_size)
                use_precond = use_L or use_R

                # ── Init ─────────────────────────────────────────────────────
                if len(state) == 0:
                    state['step']            = 0
                    state['momentum_buffer'] = g_2d.clone()
                    state['smoothed_ratio']  = (min(m, n)) ** 0.5

                    if use_precond:
                        if use_L:
                            state['L']     = eps * torch.eye(m, device=g.device, dtype=torch.float32)
                            state['eig_L'] = None
                        if use_R:
                            state['R']     = eps * torch.eye(n, device=g.device, dtype=torch.float32)
                            state['eig_R'] = None
                        state['eig_update_count'] = 0

                state['step'] += 1
                t = state['step']

                # ── Step 1: Momentum EMA ──────────────────────────────────────
                buf = state['momentum_buffer']
                if t > 1:
                    buf.mul_(momentum).add_(g_2d, alpha=1. - momentum)

                # ═════════════════════════════════════════════════════════════
                # BRANCH A — no preconditioning
                #   · 1-D parameters (m=1 after to_2d, but only if g.ndim==1)
                #   · both L and R are too large to store (use_precond=False)
                # ═════════════════════════════════════════════════════════════
                if not use_precond:
                    u = norm_backend.lmo(buf)
                    self.effective_lrs[group['norm']] = scale * lr

                # ═════════════════════════════════════════════════════════════
                # BRANCH B — Mousse-Scion preconditioning
                # ═════════════════════════════════════════════════════════════
                else:
                    # ── Step 2: Curvature EMA ─────────────────────────────────
                    update_ema = (eig_schedule is None or
                                  t <= eig_schedule.get('warmdown_start', float('inf')))
                    if update_ema:
                        if use_L:
                            state['L'].mul_(beta).add_(g_2d @ g_2d.T, alpha=1.0 - beta)
                        if use_R:
                            state['R'].mul_(beta).add_(g_2d.T @ g_2d, alpha=1.0 - beta)

                    # ── Step 3: Bias correction ───────────────────────────────
                    if LR_correction:
                        bc = 1.0 - beta ** t
                        L_hat = state['L'] / bc if use_L else None
                        R_hat = state['R'] / bc if use_R else None
                    else:
                        L_hat = state['L'] if use_L else None
                        R_hat = state['R'] if use_R else None

                    # ── Step 4: Resolve effective eig_update_freq ─────────────
                    # eig_schedule=None  → use fixed eig_update_freq unchanged.
                    # eig_schedule set   → delegate to scheduler:
                    #   returns None     → skip eigh this phase (EMA unreliable)
                    #   returns int T    → refresh if t % T == 1 or first call
                    first_eig = (use_L and state.get('eig_L') is None) or \
                                (use_R and state.get('eig_R') is None)
                    if eig_schedule is None:
                        run_eigh = (t % eig_update_freq == 1 or first_eig)
                    else:
                        effective_T = get_eig_update_freq(t, eig_schedule)
                        if effective_T is None:
                            run_eigh = False
                        else:
                            run_eigh = (t % effective_T == 1 or first_eig)

                    # ── Step 5: Eigendecomposition ────────────────────────────
                    if run_eigh:
                        if use_L:
                            if use_trace_norm:
                                trace_L = L_hat.trace().clamp(min=eps)
                                L_norm  = L_hat * (m / trace_L)
                            else:
                                L_norm = L_hat
                            eval_L, evec_L = torch.linalg.eigh(
                                L_norm + eps * torch.eye(m, device=g.device)
                            )
                            eval_L = clean_eigenvalues(eval_L, eps)
                            state['eig_L'] = (eval_L, evec_L)

                        if use_R:
                            if use_trace_norm:
                                trace_R = R_hat.trace().clamp(min=eps)
                                R_norm  = R_hat * (n / trace_R)
                            else:
                                R_norm = R_hat
                            eval_R, evec_R = torch.linalg.eigh(
                                R_norm + eps * torch.eye(n, device=g.device)
                            )
                            eval_R = clean_eigenvalues(eval_R, eps)
                            state['eig_R'] = (eval_R, evec_R)

                        state['eig_update_count'] += 1

                        # if state['eig_update_count']:
                        #     # ── ADD THIS CHECK ──
                        #     # Check if distributed is initialized. If not, it's single GPU (safe to save).
                        #     # If it is, only let rank 0 save to avoid ID mismatch and file corruption.
                        #     import torch.distributed as dist
                        #     is_master = not dist.is_initialized() or dist.get_rank() == 0
                            
                        #     if is_master:
                        #         # Create directory if it doesn't exist
                        #         save_dir = "eigenvalue_logs"
                        #         os.makedirs(save_dir, exist_ok=True)
                                
                        #         # Use the memory address of the parameter id(p) to separate layers, 
                        #         # and 't' to mark the global step.
                        #         filename = os.path.join(save_dir, f"evals_param{id(p)}_step{t}.pt")
                                
                        #         # Save as a dictionary directly to disk
                        #         torch.save({
                        #             'eval_L': eval_L.detach().cpu(),
                        #             'eval_R': eval_R.detach().cpu()
                        #         }, filename)

                    # ── Step 6: Whitening / LMO / Unwhitening ─────────────────
                    # Graceful degradation: if no eig has been computed yet (Phase 1
                    # of schedule or very first step), fall back to plain Scion.
                    eig_ready = ((not use_L or state.get('eig_L') is not None) and
                                 (not use_R or state.get('eig_R') is not None))
                    if not eig_ready:
                        u = norm_backend.lmo(buf)
                        self.effective_lrs[group['norm']] = scale * lr
                    else:
                        # Retrieve cached eigenfactors (None when that side is skipped)
                        eig_L = state.get('eig_L')   # (eval_L, evec_L) or None
                        eig_R = state.get('eig_R')   # (eval_R, evec_R) or None

                        if eig_L is not None:
                            eval_L, evec_L = eig_L
                            scale_L = eval_L.pow(alpha)   # [m]
                        if eig_R is not None:
                            eval_R, evec_R = eig_R
                            scale_R = eval_R.pow(alpha)   # [n]

                        # ── Spectral norms: use the Q^TQ = I orthogonality trick ──
                        #
                        # lmo = msign (Newton-Schulz).  Because msign is applied
                        # between two orthogonal factors:
                        #   L^{-p} msign(L^{-p} M R^{-p}) R^{-p}
                        #     = Q_L Λ_L^{-p} msign(Λ_L^{-p} Q_L^T M Q_R Λ_R^{-p}) Λ_R^{-p} Q_R^T
                        # (the Q^T Q = I cancels between the outer and inner Λ factors)
                        # so the entire computation lives in the compact eigenbasis.
                        if is_spectral:
                            # --- Whiten into eigenbasis ---
                            M_white = buf
                            if eig_L is not None:
                                M_white = evec_L.T @ M_white
                                M_white = M_white / scale_L.unsqueeze(1)
                            if eig_R is not None:
                                M_white = M_white @ evec_R
                                M_white = M_white / scale_R.unsqueeze(0)

                            # --- LMO in (compact) whitened space ---
                            u = norm_backend.lmo(M_white)

                            fro_norm  = M_white.norm()
                            dual_norm = (u * M_white).sum()
                            current_ratio = dual_norm / fro_norm.clamp(min=eps)
                            state['smoothed_ratio'] = (
                                beta_scale * state['smoothed_ratio']
                                + (1.0 - beta_scale) * current_ratio
                            )
                            norm_ratio = state['smoothed_ratio']

                            # --- Graft reference norm ---
                            if apply_grafting == "fro":
                                graft_norm = fro_norm
                            elif apply_grafting == "lmo":
                                graft_norm = u.norm()
                            elif apply_grafting == "ratio":
                                graft_norm = norm_ratio
                            elif apply_grafting == "interpolate":
                                warmup_steps = group.get('norm_warmup_steps', 500.)
                                tau_k = min(1.0, t / warmup_steps)
                                graft_norm = (1. - tau_k) * fro_norm + tau_k * dual_norm
                            else:  # "dual"
                                graft_norm = dual_norm

                            # --- Unwhiten: Λ^{-p} u Λ^{-p}, then rotate back ---
                            if eig_R is not None:
                                u = u / scale_R.unsqueeze(0)
                                u = u @ evec_R.T
                            if eig_L is not None:
                                u = u / scale_L.unsqueeze(1)
                                u = evec_L @ u

                        # ── Non-spectral norms: rebuild M_tilde in the original ──
                        #    space, because lmo(L^{-p} M R^{-p}) ≠ lmo in eigenbasis
                        #    for ColNorm, RowNorm, Sign, etc.
                        #
                        else:
                            # --- Build M_tilde in the original parameter space ---
                            M_tilde = buf
                            if eig_L is not None:
                                # Apply L^{-p} on the left: Q_L Λ_L^{-p} Q_L^T M
                                M_tilde = evec_L @ (evec_L.T @ M_tilde / scale_L.unsqueeze(1))
                            if eig_R is not None:
                                # Apply R^{-p} on the right: (...) Q_R Λ_R^{-p} Q_R^T
                                M_tilde = (M_tilde @ evec_R / scale_R.unsqueeze(0)) @ evec_R.T

                            # --- LMO in original space ---
                            u = norm_backend.lmo(M_tilde)

                            fro_norm  = M_tilde.norm()
                            dual_norm = (u * M_tilde).sum()
                            # For non-spectral norms, use the instantaneous ratio
                            # directly — no EMA smoothing (smoothed_ratio is only
                            # updated in the Spectral branch where it is well-motivated).
                            norm_ratio = 1.#dual_norm / fro_norm.clamp(min=eps)

                            # --- Graft reference norm ---
                            if apply_grafting == "fro":
                                graft_norm = fro_norm
                            elif apply_grafting == "lmo":
                                graft_norm = u.norm()
                            elif apply_grafting == "ratio":
                                graft_norm = norm_ratio
                            elif apply_grafting == "interpolate":
                                warmup_steps = group.get('norm_warmup_steps', 500.)
                                tau_k = min(1.0, t / warmup_steps)
                                graft_norm = (1. - tau_k) * fro_norm + tau_k * dual_norm
                            else:  # "dual"
                                graft_norm = dual_norm

                            # --- Apply L^{-p} and R^{-p} to the lmo output ---
                            if eig_L is not None:
                                u = evec_L @ (evec_L.T @ u / scale_L.unsqueeze(1))
                            if eig_R is not None:
                                u = (u @ evec_R / scale_R.unsqueeze(0)) @ evec_R.T

                        # ── Graft (common to both spectral and non-spectral) ──────
                        u_norm = u.norm()
                        if u_norm > eps:
                            u = (graft_norm / u_norm) * u

                        self.effective_lrs[group['norm']] = lr * scale * graft_norm / u_norm
                        self.fro_norms[group['norm']]   = fro_norm.item()  if hasattr(fro_norm,  'item') else fro_norm
                        self.dual_norms[group['norm']]  = dual_norm.item() if hasattr(dual_norm, 'item') else dual_norm
                        self.denom_norms[group['norm']] = u_norm.item()    if hasattr(u_norm,    'item') else u_norm
                        self.norm_ratios[group['norm']] = norm_ratio.item() if hasattr(norm_ratio,'item') else norm_ratio

                # ── Step 7: Parameter update ──────────────────────────────────
                # w_{t+1} = (1 - η) w_t - η · s · u
                update = (scale * u).reshape(g.shape)
                if not unconstrained:
                    p.data.mul_(1.0 - lr)
                p.data.add_(update, alpha=-lr)

        return loss

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            scale        = group['scale']
            for p in group['params']:
                norm_backend.init(p)
                p.data *= scale