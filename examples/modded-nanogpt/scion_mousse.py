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

    Four-phase schedule designed to minimise eigh calls while preserving
    the preconditioning quality that matters most:

      Phase 1  [0, eig_warmup_steps):
          Returns None  →  skip eigh entirely.

      Phase 2a  [eig_warmup_steps, stable_start):
          Returns T_init  →  frequent refreshes (default 10).

      Phase 2b  [stable_start, warmdown_start]:
          Linearly interpolates T from T_init to T_mid.

      Phase 3  (warmdown_start, total_steps]:
          Linearly interpolates T from T_mid to T_warmdown.

    Args:
        t (int):             Current global training step.
        eig_schedule (dict): Must contain:
            'eig_warmup_steps' (int, default 0)  — end of Phase 1.
            'stable_start'     (int)              — end of Phase 2a /
                                                    start of Phase 2b ramp.
                                                    Rule of thumb: ~3/(1-β).
                                                    With β=0.99 → ~300 steps.
            'warmdown_start'   (int)              — end of Phase 2b /
                                                    start of Phase 3.
            'total_steps'      (int)              — end of Phase 3.
            'T_init'           (int)              — freq during Phase 2a.
            'T_mid'            (int)              — freq at start of Phase 3
                                                    (= end of Phase 2b ramp).
            'T_warmdown'       (int)              — freq at end of Phase 3.

    Returns:
        int | None: Effective eig_update_freq for step t, or None to skip.
    """
    eig_warmup     = eig_schedule.get('eig_warmup_steps', 0)
    stable_start   = eig_schedule['stable_start']
    warmdown_start = eig_schedule['warmdown_start']
    total_steps    = eig_schedule['total_steps']
    T_init         = eig_schedule['T_init']
    T_mid          = eig_schedule['T_mid']
    T_warmdown     = eig_schedule['T_warmdown']

    # Phase 1 — EMA not yet reliable, skip eigh entirely
    if t < eig_warmup:
        return None

    # Phase 2a — EMA warming up, refresh frequently
    if t < stable_start:
        return T_init

    # Phase 2b — EMA converged, ramp T_init → T_mid to save cost
    if t <= warmdown_start:
        stable_len = max(warmdown_start - stable_start, 1)
        progress   = min((t - stable_start) / stable_len, 1.0)
        T = T_init + progress * (T_mid - T_init)
        return max(1, int(round(T)))

    # Phase 3 — lr warmdown, ramp T_mid → T_warmdown aggressively
    warmdown_len = max(total_steps - warmdown_start, 1)
    progress     = min((t - warmdown_start) / warmdown_len, 1.0)
    T = T_mid + progress * (T_warmdown - T_mid)
    return max(1, int(round(T)))


#######################################################
# MousseScion
#######################################################


class MousseScion(torch.optim.Optimizer):
    """
    Mousse-style L,R preconditioning applied to the Scion optimizer.

    Update equations (per step t):
        m_t  =  (1-μ) · m_{t-1}  +  μ · G_t                     [momentum]

    When skip_preconditioning is False (full Mousse-Scion path):
        L_t  =  β · L_{t-1}  +  (1-β) · G_t G_t^T               [left  curvature EMA]
        R_t  =  β · R_{t-1}  +  (1-β) · G_t^T G_t               [right curvature EMA]
        (Λ_L, Q_L) = eigh(L̂_t),  (Λ_R, Q_R) = eigh(R̂_t)       [every effective T steps]
        M̃      =  Q_L^T  m_t  Q_R                                [whiten: rotate]
        M̃_{ij} /= λ_i^(L,α) · λ_j^(R,α)                        [whiten: scale]
        u      =  lmo(M̃)
        n*     =  ‖u‖_F  or  ⟨u, M̃⟩                            [graft reference]
        u_{ij} /= λ_i^(L,α) · λ_j^(R,α)                        [unwhiten: scale]
        u      =  Q_L  u  Q_R^T                                  [unwhiten: rotate]
        u      ←  (n* / ‖u‖_F) · u                              [graft norm]

    When skip_preconditioning is True (Sign / large-vocab layers):
        u  =  lmo(m_t)   [exact for Sign; sign(P M Q) = sign(M) for PD P,Q]

    Frank-Wolfe update (both paths):
        w_{t+1}  =  (1 - η) · w_t  -  η · s · u

    Args:
        params:                   Parameters to optimize.
        lr (float):               Learning rate η (default: 1e-3).
        momentum (float):         Momentum EMA coefficient (default: 0.9).
        norm (str):               LMO norm class (default: 'Auto').
        norm_kwargs (dict):       Extra kwargs for the norm class (default: {}).
        scale (float):            Constraint radius s (default: 1.0).
        unconstrained (bool):     Skip (1-lr) shrinkage (default: False).
        beta (float):             Curvature EMA decay (default: 0.999).
        alpha (float):            Curvature exponent (default: 0.125).
        eps (float):              Eigenvalue damping (default: 1e-8).
        eig_update_freq (int):    Fixed eigh frequency. Used only when
                                  eig_schedule is None (default: 10).
        eig_schedule (dict|None): Three-phase frequency schedule. When set,
                                  eig_update_freq is ignored. See
                                  get_eig_update_freq() for full key docs.
                                  Example for a 9750-step run:
                                    {
                                      'eig_warmup_steps': 200,
                                      'warmdown_start':   7500,
                                      'total_steps':      9750,
                                      'T_train':          10,
                                      'T_warmdown':       200,
                                    }
        use_trace_normalization (bool): Trace-normalise L,R before eigh (default: True).
        LR_correction (bool):     Bias-correct curvature EMAs (default: True).
        apply_grafting (str):     'fro' (default) or 'dual'.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        norm: str = 'Auto',
        norm_kwargs: dict = None,
        scale: float = 1.0,
        unconstrained: bool = False,
        beta: float = 0.999,
        alpha: float = 0.125,
        eps: float = 1e-8,
        eig_update_freq: int = 10,
        eig_schedule: dict | None = None,
        use_trace_normalization: bool = True,
        LR_correction: bool = True,
        apply_grafting: str = "fro",
        norm_warmup_steps: int= 500,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0,1], got {momentum}.")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0,1), got {beta}.")
        if norm not in norm_dict:
            raise ValueError(f"Unknown norm '{norm}'. Choose from {list(norm_dict.keys())}.")
        if apply_grafting not in ("fro", "dual", "interpolate"):
            raise ValueError(f"apply_grafting must be 'fro' or 'dual', got '{apply_grafting}'.")
        if norm_kwargs is None:
            norm_kwargs = {}
        if eig_schedule is not None:
            for key in ('stable_start', 'warmdown_start', 'total_steps',
                        'T_init', 'T_mid', 'T_warmdown'):
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
        )
        super().__init__(params, defaults)
        self.effective_lrs = {}
        self.fro_norms = {}
        self.dual_norms = {}
        self.denom_norms = {}

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr              = group['lr']
            momentum        = group['momentum']
            scale           = group['scale']
            unconstrained   = group['unconstrained']
            norm_backend    = norm_dict[group['norm']](**group['norm_kwargs'])
            beta            = group['beta']
            alpha           = group['alpha']
            eps             = group['eps']
            eig_update_freq = group['eig_update_freq']
            eig_schedule    = group['eig_schedule']
            use_trace_norm  = group['use_trace_normalization']
            LR_correction   = group['LR_correction']
            apply_grafting  = group['apply_grafting']
            skip_precond    = group['norm'] == 'Sign'

            for p in group['params']:
                if p.grad is None:
                    continue

                g    = p.grad
                g_2d = to_2d(g).float()
                m, n = g_2d.shape
                state = self.state[p]

                # ── Init ─────────────────────────────────────────────────────
                if len(state) == 0:
                    state['step']            = 0
                    state['momentum_buffer'] = g_2d.clone()
                    if not skip_precond:
                        state['L']     = eps * torch.eye(m, device=g.device, dtype=torch.float32)
                        state['R']     = eps * torch.eye(n, device=g.device, dtype=torch.float32)
                        state['eig_L'] = None
                        state['eig_R'] = None
                        state['eig_update_count'] = 0

                state['step'] += 1
                t = state['step']

                # ── Step 1: Momentum EMA ──────────────────────────────────────
                buf = state['momentum_buffer']
                if t > 1:
                    buf.mul_(momentum).add_(g_2d, alpha=1. - momentum)

                # ═════════════════════════════════════════════════════════════
                # BRANCH A — skip preconditioning (Sign / large-vocab layers)
                # ═════════════════════════════════════════════════════════════
                if skip_precond:
                    u = norm_backend.lmo(buf)
                    self.effective_lrs[group['norm']] = scale * lr

                # ═════════════════════════════════════════════════════════════
                # BRANCH B — full Mousse-Scion preconditioning
                # ═════════════════════════════════════════════════════════════
                else:
                    # ── Step 2: Curvature EMA ─────────────────────────────────
                    state['L'].mul_(beta).add_(g_2d @ g_2d.T, alpha=1.0 - beta)
                    state['R'].mul_(beta).add_(g_2d.T @ g_2d, alpha=1.0 - beta)

                    # ── Step 3: Bias correction ───────────────────────────────
                    if LR_correction:
                        bc    = 1.0 - beta ** t
                        L_hat = state['L'] / bc
                        R_hat = state['R'] / bc
                    else:
                        L_hat = state['L']
                        R_hat = state['R']

                    # ── Step 4: Resolve effective eig_update_freq ─────────────
                    # eig_schedule=None  → use fixed eig_update_freq unchanged.
                    # eig_schedule set   → delegate to scheduler:
                    #   returns None     → skip eigh this phase (EMA unreliable)
                    #   returns int T    → refresh if t % T == 1 or first call
                    if eig_schedule is None:
                        run_eigh = (t % eig_update_freq == 1 or state['eig_L'] is None)
                    else:
                        effective_T = get_eig_update_freq(t, eig_schedule)
                        if effective_T is None:
                            run_eigh = False   # Phase 1: skip entirely
                        else:
                            run_eigh = (t % effective_T == 1 or state['eig_L'] is None)

                    # ── Step 5: Eigendecomposition ────────────────────────────
                    if run_eigh:
                        if use_trace_norm:
                            trace_L = L_hat.trace().clamp(min=eps)
                            trace_R = R_hat.trace().clamp(min=eps)
                            L_norm  = L_hat * (m / trace_L)
                            R_norm  = R_hat * (n / trace_R)
                        else:
                            L_norm = L_hat
                            R_norm = R_hat

                        eval_L, evec_L = torch.linalg.eigh(
                            L_norm + eps * torch.eye(m, device=g.device)
                        )
                        eval_R, evec_R = torch.linalg.eigh(
                            R_norm + eps * torch.eye(n, device=g.device)
                        )
                        eval_L = clean_eigenvalues(eval_L, eps)
                        eval_R = clean_eigenvalues(eval_R, eps)

                        state['eig_L'] = (eval_L, evec_L)
                        state['eig_R'] = (eval_R, evec_R)

                        state['eig_update_count'] += 1

                        if state['eig_update_count'] % 5 == 0:
                            # Create directory if it doesn't exist
                            save_dir = "eigenvalue_logs"
                            os.makedirs(save_dir, exist_ok=True)
                            
                            # Use the memory address of the parameter id(p) to separate layers, 
                            # and 't' to mark the global step.
                            filename = os.path.join(save_dir, f"evals_param{id(p)}_step{t}.pt")
                            
                            # Save as a dictionary directly to disk
                            torch.save({
                                'eval_L': eval_L.detach().cpu(),
                                'eval_R': eval_R.detach().cpu()
                            }, filename)

                    # ── Step 6: Whitening / LMO / Unwhitening ─────────────────
                    # If eig_L is still None (Phase 1 of schedule, first step),
                    # fall back to plain Scion for this step — graceful degradation.
                    if state['eig_L'] is None:
                        u = norm_backend.lmo(buf)
                        self.effective_lrs[group['norm']] = scale * lr
                    else:
                        eval_L, evec_L = state['eig_L']
                        eval_R, evec_R = state['eig_R']

                        scale_L = eval_L.pow(alpha)   # [m]
                        scale_R = eval_R.pow(alpha)   # [n]

                        # Whiten
                        M_white = evec_L.T @ buf @ evec_R
                        M_white = M_white / scale_L.unsqueeze(1)
                        M_white = M_white / scale_R.unsqueeze(0)

                        # LMO in whitened space
                        u = norm_backend.lmo(M_white)

                        fro_norm = u.norm()
                        dual_norm = (u * M_white).sum() #/ (min(m, n) ** 0.5)

                        # Graft reference norm
                        if apply_grafting == "fro":
                            graft_norm = fro_norm
                        elif apply_grafting == "interpolate":
                            warmup_steps = group.get('norm_warmup_steps', 500.)
                            tau_k = min(1.0, t / warmup_steps)
                            graft_norm = (1. - tau_k) * fro_norm + tau_k * dual_norm

                        else:  # "dual"
                            graft_norm = dual_norm

                        # Unwhiten
                        u = u / scale_L.unsqueeze(1)
                        u = u / scale_R.unsqueeze(0)
                        u = evec_L @ u @ evec_R.T

                        # Graft
                        u_norm = u.norm()
                        if u_norm > eps:
                            u = (graft_norm / u_norm) * u

                        self.effective_lrs[group['norm']] = lr * scale * graft_norm / u_norm
                        self.fro_norms[group['norm']] = fro_norm.item() if hasattr(fro_norm, 'item') else fro_norm
                        self.dual_norms[group['norm']] = dual_norm.item() if hasattr(dual_norm, 'item') else dual_norm
                        self.denom_norms[group['norm']] = u_norm.item() if hasattr(u_norm, 'item') else u_norm

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