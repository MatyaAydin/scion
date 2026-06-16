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
          Returns None  ->  no eigenbasis exists yet, step() falls back to
          plain (unpreconditioned) Scion.

      Phase 2  [eig_warmup_steps, warmdown_start):
          Returns T_init  ->  frequent refreshes (e.g. 125).

      Phase 3  [warmdown_start, total_steps]:
          Returns None  ->  stop refreshing the eigenbasis. step() keeps
          reusing whatever L/R eigenbasis was last computed during Phase 2
          instead of discarding it, so preconditioning stays on through
          warmdown -- it just stops being updated.

    Args:
        t (int):             Current global training step.
        eig_schedule (dict): Must contain:
            'eig_warmup_steps' (int)  -- end of Phase 1.
            'warmdown_start'   (int)  -- end of Phase 2 / start of Phase 3.
            'T_init'           (int)  -- freq during Phase 2.

    Returns:
        int | None: Effective eig_update_freq for step t, or None to skip
        refreshing the eigenbasis this step.
    """
    eig_warmup     = eig_schedule.get('eig_warmup_steps', 500)
    warmdown_start = eig_schedule.get('warmdown_start', 5250)
    T_init         = eig_schedule.get('T_init', 125)

    if t < eig_warmup:
        return None

    if t < warmdown_start:
        return T_init

    return None


#######################################################
# MousseScion
#######################################################


class MousseScion(torch.optim.Optimizer):
    """
    Mousse-style L,R preconditioning applied to the Scion optimizer.

    Update equations (per step t):
        m_t  =  (1-mu) . m_{t-1}  +  mu . G_t                     [momentum]

    When skip_preconditioning is False (full Mousse-Scion path):
        L_t  =  beta . L_{t-1}  +  (1-beta) . G_t G_t^T           [left  curvature EMA]
        R_t  =  beta . R_{t-1}  +  (1-beta) . G_t^T G_t           [right curvature EMA]
        (Lambda_L, Q_L) = eigh(L_hat_t),  (Lambda_R, Q_R) = eigh(R_hat_t)  [every effective T steps]
        M~      =  Q_L^T  m_t  Q_R                                [whiten: rotate]
        M~_{ij} /= lambda_i^(L,alpha) . lambda_j^(R,alpha)        [whiten: scale]
        u      =  lmo(M~)
        n*     =  ||u||_F  or  <u, M~>                            [graft reference]
        u_{ij} /= lambda_i^(L,alpha) . lambda_j^(R,alpha)        [unwhiten: scale]
        u      =  Q_L  u  Q_R^T                                  [unwhiten: rotate]
        u      <-  (n* / ||u||_F) . u                            [graft norm]

    When skip_preconditioning is True (Sign / large-vocab layers):
        u  =  lmo(m_t)   [exact for Sign; sign(P M Q) = sign(M) for PD P,Q]

    Frank-Wolfe update (both paths):
        w_{t+1}  =  (1 - eta) . w_t  -  eta . s . u

    Args:
        params:                   Parameters to optimize.
        lr (float):               Learning rate eta (default: 1e-3).
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
        apply_grafting (str):     'fro' (default) or 'dual'.
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
            beta_scale      = group['beta_scale']
            skip_precond    = (group['norm'] != 'Spectral') and (group['norm'] != 'SpectralConv')

            params = [p for p in group['params'] if p.grad is not None]
            if not params:
                continue

            # ── Step 1: per-parameter bookkeeping ──────────────────────────
            # Init state if needed, bump the step counter, reshape each
            # gradient to 2D once. Inherently per-parameter (shapes/state
            # differ) but cheap -- no linear algebra happens here.
            grads_2d = []
            bufs     = []
            ts       = []
            for p in params:
                g    = p.grad
                g_2d = to_2d(g).float()
                m, n = g_2d.shape
                state = self.state[p]

                if len(state) == 0:
                    state['step']            = 0
                    state['momentum_buffer'] = g_2d.clone()

                    state['smoothed_ratio']  = (min(m, n)) ** 0.5
                    if not skip_precond:
                        state['L']     = eps * torch.eye(m, device=g.device, dtype=torch.float32)
                        state['R']     = eps * torch.eye(n, device=g.device, dtype=torch.float32)
                        state['eig_L'] = None
                        state['eig_R'] = None
                        state['eig_update_count'] = 0

                state['step'] += 1
                grads_2d.append(g_2d)
                bufs.append(state['momentum_buffer'])
                ts.append(state['step'])

            # ── Step 2: Momentum EMA, batched across the whole group ───────
            # [CHANGE 2] Replaces the per-parameter buf.mul_().add_() calls
            # with a single pair of multi-tensor (foreach) kernel launches
            # covering every parameter in the group at once. Params on their
            # very first step already have buf == g_2d (set above at init),
            # so they're excluded here to avoid double-counting that first
            # gradient -- identical semantics to the old `if t > 1:` guard.
            ema_idx = [i for i, t in enumerate(ts) if t > 1]
            if ema_idx:
                ema_bufs  = [bufs[i] for i in ema_idx]
                ema_grads = [grads_2d[i] for i in ema_idx]
                torch._foreach_mul_(ema_bufs, momentum)
                torch._foreach_add_(ema_bufs, ema_grads, alpha=1.0 - momentum)

            updates = []

            for p, g_2d, buf, t in zip(params, grads_2d, bufs, ts):
                g    = p.grad
                m, n = g_2d.shape
                state = self.state[p]

                # ═════════════════════════════════════════════════════════
                # BRANCH A — skip preconditioning (Sign / large-vocab layers)
                # ═════════════════════════════════════════════════════════
                if skip_precond:
                    u = norm_backend.lmo(buf)
                    self.effective_lrs[group['norm']] = scale * lr

                # ═════════════════════════════════════════════════════════
                # BRANCH B — full Mousse-Scion preconditioning
                # ═════════════════════════════════════════════════════════
                else:
                    # ── Step 3: Curvature EMA ─────────────────────────────
                    state['L'].mul_(beta).add_(g_2d @ g_2d.T, alpha=1.0 - beta)
                    state['R'].mul_(beta).add_(g_2d.T @ g_2d, alpha=1.0 - beta)

                    # ── Step 4: Resolve effective eig_update_freq ─────────
                    # eig_schedule=None  -> use fixed eig_update_freq unchanged.
                    # eig_schedule set   -> delegate to scheduler:
                    #   returns None     -> skip refreshing the eigenbasis
                    #   returns int T    -> refresh if t % T == 1 or first call
                    if eig_schedule is None:
                        run_eigh = (t % eig_update_freq == 1 or state['eig_L'] is None)
                    else:
                        effective_T = get_eig_update_freq(t, eig_schedule)
                        if effective_T is None:
                            run_eigh = False
                            # Deliberately NOT resetting state['eig_L']/state['eig_R']
                            # here. During warmup they're already None and step 6
                            # falls back to plain Scion below; during warmdown they
                            # hold the last eigenbasis from Phase 2, and we want
                            # step 6 to keep reusing it instead of discarding it.
                        else:
                            run_eigh = (t % effective_T == 1 or state['eig_L'] is None)

                    # ── Step 5: Eigendecomposition ────────────────────────
                    # [CHANGE 1] Bias correction and trace normalization are
                    # only ever read inside this block, so they're computed
                    # lazily here instead of unconditionally every step.
                    if run_eigh:
                        if LR_correction:
                            bc    = 1.0 - beta ** t
                            L_hat = state['L'] / bc
                            R_hat = state['R'] / bc
                        else:
                            L_hat = state['L']
                            R_hat = state['R']

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

                    # ── Step 6: Whitening / LMO / Unwhitening ─────────────
                    # If eig_L is still None (Phase 1, before any eigenbasis
                    # has ever been computed), fall back to plain Scion.
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

                        fro_norm = M_white.norm()
                        dual_norm = (u * M_white).sum()
                        current_ratio = dual_norm / fro_norm.clamp(min=eps)

                        state['smoothed_ratio'] = beta_scale * state['smoothed_ratio'] + (1.0 - beta_scale) * current_ratio
                        norm_ratio = state['smoothed_ratio']

                        # Graft reference norm
                        if apply_grafting == "fro":
                            graft_norm = fro_norm
                        elif apply_grafting == "lmo":  # is actually equal to sqrt(d)
                            graft_norm = u.norm()
                        elif apply_grafting == "ratio":
                            graft_norm = norm_ratio
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
                        self.norm_ratios[group['norm']] = norm_ratio.item() if hasattr(norm_ratio, 'item') else norm_ratio

                updates.append((scale * u).reshape(g.shape))

            # ── Step 7: Parameter update, batched across the whole group ───
            # [CHANGE 2] w_{t+1} = (1 - eta) w_t - eta . s . u, applied to
            # every parameter in the group via two foreach calls instead of
            # a per-parameter mul_()/add_() pair.
            param_data = [p.data for p in params]
            if not unconstrained:
                torch._foreach_mul_(param_data, 1.0 - lr)
            torch._foreach_add_(param_data, updates, alpha=-lr)

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