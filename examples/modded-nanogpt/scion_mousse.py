import math
import torch


#######################################################
# Norm classes (unchanged from ScionShampoo)
#######################################################


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError


class ColNorm(Norm):
    """
    Column-wise normalization.

    Args:
        normalized (bool): If True, normalizes by the input dimension.
        transpose (bool): If True, transposes input before normalization (use for embedding layers).
    """
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
    """
    Row-wise normalization.

    Args:
        normalized (bool): If True, normalizes by the input dimension.
        transpose (bool): If True, transposes input before normalization (use for embedding layers).
    """
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
        if g.ndim == 3:    # Conv1d
            out_channels, in_channels, k = g.shape
            g *= (out_channels / in_channels)**0.5 / k
        elif g.ndim == 4:   # Conv2d
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
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    From: https://github.com/KellerJordan/modded-nanogpt
    """
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
    """Reshape gradient to 2D: [first_dim, rest_flattened]."""
    if g.dim() == 1:
        return g.unsqueeze(0)       # [1, d]
    return g.reshape(g.shape[0], -1)  # [d0, d1*d2*...]


def clean_eigenvalues(evals, epsilon):
    """
    λ := λ - λ_min + ε
    Shift all eigenvalues so the smallest is at least ε.
    """
    min_eig = evals.min()
    shift = torch.clamp(-min_eig, min=0.0) + epsilon
    return evals + shift


#######################################################
# MousseScion
#######################################################


class MousseScion(torch.optim.Optimizer):
    """
    Mousse-style L,R preconditioning applied to the Scion optimizer.
 
    Instead of materializing full L^{-p} and R^{-p} matrices (as ScionShampoo does),
    this optimizer maintains explicit eigendecompositions of L and R and applies
    whitening/unwhitening via eigenvector rotations and per-eigenvalue scaling.
    After the LMO step the Frobenius norm is saved and used to graft the scale
    back onto the unwhitened update, preventing the round-trip from distorting
    the update magnitude.
 
    Update equations (per step t):
        m_t  =  μ · m_{t-1}  +  (1-μ) · G_t                     [momentum]
 
    When skip_preconditioning is False (full Mousse-Scion path):
        L_t  =  β · L_{t-1}  +  (1-β) · G_t G_t^T               [left  curvature EMA]
        R_t  =  β · R_{t-1}  +  (1-β) · G_t^T G_t               [right curvature EMA]
        (Λ_L, Q_L) = eigh(L̂_t),   (Λ_R, Q_R) = eigh(R̂_t)      [every eig_update_freq steps]
        M̃      =  Q_L^T  m_t  Q_R                                [whiten: rotate]
        M̃_{ij} /= λ_i^(L,α) · λ_j^(R,α)                        [whiten: scale]
        u      =  lmo(M̃)
        n*     =  ‖u‖_F                                          [graft reference norm]
        ρ      =  ⟨u, M̃⟩                                        [dual norm in whitened space]
        u_{ij} /= λ_i^(L,α) · λ_j^(R,α)                        [unwhiten: scale]
        u      =  Q_L  u  Q_R^T                                  [unwhiten: rotate]
        u      ←  (n* / ‖u‖_F) · u                              [graft norm]
 
    When skip_preconditioning is True (memory-efficient path for Sign / large vocab layers):
        L, R, eigh and all whitening/unwhitening ops are skipped entirely.
        u      =  lmo(m_t)                                        [LMO directly on momentum]
        ρ      =  ⟨u, m_t⟩                                       [dual norm in original space]
        Grafting is a no-op (no round-trip distortion to correct).
 
        Correctness note: for Sign specifically this path is *exact*, not an approximation.
        sign(L^{-α} M R^{-α}) / d_in  =  sign(M) / d_in  =  lmo_Sign(M)
        because L^{-α} and R^{-α} are positive-definite and cannot flip signs.
        For other norms (e.g. ColNorm with rows rescaled by column statistics)
        skip_preconditioning is a deliberate memory/accuracy trade-off.
 
    Frank-Wolfe parameter update (both paths):
        w_{t+1}  =  (1 - η) · w_t  -  η · s · ρ · u
 
    Args:
        params:                   Parameters to optimize.
        lr (float):               Learning rate η (default: 1e-3).
        momentum (float):         EMA coefficient for momentum buffer, i.e. (1 - traditional β₁).
                                  momentum=0.1 ↔ traditional β₁=0.9 (default: 0.1).
        norm (str):               LMO norm class: 'Auto' | 'Spectral' | 'SpectralConv' |
                                  'ColNorm' | 'RowNorm' | 'BiasRMS' | 'Sign' (default: 'Auto').
        norm_kwargs (dict):       Extra kwargs forwarded to the norm class (default: {}).
        scale (float):            Constraint set radius s (default: 1.0).
        unconstrained (bool):     If True, skip the (1-lr) weight shrinkage (default: False).
        beta (float):             EMA decay for curvature matrices L and R (default: 0.999).
        alpha (float):            Curvature exponent. Whitening scales by λ^(-α) per side.
                                  α=0.5 → full Shampoo; α=0 → no preconditioning (default: 0.125).
        eps (float):              Damping added to eigenvalues and curvature init (default: 1e-8).
        eig_update_freq (int):    Recompute eigendecompositions every this many steps (default: 10).
        use_trace_normalization (bool): Normalize L and R by their traces before decomposition,
                                  making eigenvalue magnitudes scale-invariant (default: True).
        LR_correction (bool):     Apply bias correction to L and R EMAs (default: True).
        use_dual_norm (bool):     Multiply update by ρ = ⟨lmo(·), ·⟩ (duality gap).
                                  Set False to get a pure direction-only update (default: True).
        apply_grafting (bool):    After unwhitening rescale u to match the pre-unwhiten LMO norm.
                                  No effect when skip_preconditioning=True (default: True).
        skip_preconditioning (bool | None):
                                  If True, bypass L/R curvature tracking and all
                                  whitening/unwhitening ops for this parameter group — the LMO
                                  is applied directly to the momentum buffer.
                                  If None (default), auto-set to True when norm == 'Sign',
                                  False otherwise. Set explicitly to override auto-detection.
                                  Use True for any layer whose weight matrix would produce an
                                  OOM-inducing m×m or n×n curvature matrix (e.g. lm_head with
                                  vocab_size rows/cols).
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
        use_trace_normalization: bool = True,
        LR_correction: bool = True,
        use_dual_norm: bool = True,
        apply_grafting: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}. Expected in [0, 1].")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta}. Expected in [0, 1).")
        if norm not in norm_dict:
            raise ValueError(f"Unknown norm '{norm}'. Choose from {list(norm_dict.keys())}.")
        if norm_kwargs is None:
            norm_kwargs = {}
 
        # skip_preconditioning=None means "auto": resolved per group in step()
        # based on norm name. Storing None lets per-group overrides work naturally.
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
            use_trace_normalization=use_trace_normalization,
            LR_correction=LR_correction,
            use_dual_norm=use_dual_norm,
            apply_grafting=apply_grafting,
        )
        super().__init__(params, defaults)
 
    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
 
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            lr                   = group['lr']
            momentum             = group['momentum']
            scale                = group['scale']
            unconstrained        = group['unconstrained']
            norm_backend         = norm_dict[group['norm']](**group['norm_kwargs'])
            beta                 = group['beta']
            alpha                = group['alpha']
            eps                  = group['eps']
            eig_update_freq      = group['eig_update_freq']
            use_trace_norm       = group['use_trace_normalization']
            LR_correction        = group['LR_correction']
            use_dual_norm        = group['use_dual_norm']
            apply_grafting       = group['apply_grafting']
 
            # ── Resolve skip_preconditioning for this group ──────────────────
            # None  → auto: True iff norm is 'Sign' (preconditioning is a
            #         provable no-op for Sign and the curvature matrices would
            #         be vocab_size × vocab_size — an OOM risk).
            # True  → always skip (useful for any OOM-prone large layer).
            # False → always use full Mousse-Scion preconditioning.
            skip_precond = group['norm'] == 'Sign'
 
            for p in group['params']:
                if p.grad is None:
                    continue
 
                g     = p.grad
                g_2d  = to_2d(g).float()   # always [m, n]
                m, n  = g_2d.shape
                state = self.state[p]
 
                # ── Init ─────────────────────────────────────────────────────
                if len(state) == 0:
                    state['step']            = 0
                    state['momentum_buffer'] = g_2d.clone()
                    if not skip_precond:
                        # Curvature matrices initialised to ε·I so early steps
                        # produce near-identity preconditioning rather than NaNs.
                        state['L']     = eps * torch.eye(m, device=g.device, dtype=torch.float32)
                        state['R']     = eps * torch.eye(n, device=g.device, dtype=torch.float32)
                        state['eig_L'] = None   # (eval_L, evec_L) once computed
                        state['eig_R'] = None
 
                state['step'] += 1
                t = state['step']
 
                # ── Step 1: Momentum EMA ──────────────────────────────────────
                # m_t = (1-μ) · m_{t-1}  +  μ · G_t
                buf = state['momentum_buffer']
                if t > 1:
                    buf.mul_(momentum).add_(g_2d, alpha=1. - momentum)
                # On t=1 buf was initialised to g_2d, consistent with Scion.
 
                # ═════════════════════════════════════════════════════════════
                # BRANCH A — skip preconditioning (Sign / large-vocab layers)
                # LMO is applied directly to the momentum buffer.
                # For Sign this is exact: sign(L^{-α} M R^{-α}) = sign(M).
                # For other norms it is a deliberate memory/accuracy trade-off.
                # ═════════════════════════════════════════════════════════════
                if skip_precond:
                    # ── Step 2s: LMO on raw momentum ─────────────────────────
                    u = norm_backend.lmo(buf)
 
                    # ── Step 3s: Dual norm in original space ──────────────────
                    # ρ = ⟨lmo(m_t), m_t⟩
                    if use_dual_norm:
                        rho = (u * buf).sum()
                    else:
                        rho = torch.tensor(1.0, device=g.device, dtype=torch.float32)
 
                    # No grafting needed — no eigenvalue round-trip distortion.
 
                # ═════════════════════════════════════════════════════════════
                # BRANCH B — full Mousse-Scion preconditioning
                # ═════════════════════════════════════════════════════════════
                else:
                    # ── Step 2: Curvature EMA ─────────────────────────────────
                    # L_t = β L_{t-1} + (1-β) G_t G_t^T
                    # R_t = β R_{t-1} + (1-β) G_t^T G_t
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
 
                    # ── Step 4: Eigendecomposition (every eig_update_freq steps)
                    if t % eig_update_freq == 1 or state['eig_L'] is None:
                        # Optional trace normalisation: makes eigenvalue magnitudes
                        # independent of the gradient norm scale.
                        if use_trace_norm:
                            trace_L = L_hat.trace().clamp(min=eps)
                            trace_R = R_hat.trace().clamp(min=eps)
                            L_norm  = L_hat * (m / trace_L)
                            R_norm  = R_hat * (n / trace_R)
                        else:
                            L_norm  = L_hat
                            R_norm  = R_hat
 
                        # eigh returns eigenvalues in ascending order
                        eval_L, evec_L = torch.linalg.eigh(
                            L_norm + eps * torch.eye(m, device=g.device)
                        )
                        eval_R, evec_R = torch.linalg.eigh(
                            R_norm + eps * torch.eye(n, device=g.device)
                        )
 
                        # Shift any negative eigenvalues (numerical noise) to ≥ eps
                        eval_L = clean_eigenvalues(eval_L, eps)
                        eval_R = clean_eigenvalues(eval_R, eps)
 
                        state['eig_L'] = (eval_L, evec_L)
                        state['eig_R'] = (eval_R, evec_R)
 
                    eval_L, evec_L = state['eig_L']
                    eval_R, evec_R = state['eig_R']
 
                    # Precompute per-eigenvalue scale factors λ^α (reused for
                    # both whitening and unwhitening passes)
                    scale_L = eval_L.pow(alpha)   # [m]
                    scale_R = eval_R.pow(alpha)   # [n]
 
                    # ── Step 5: Whitening ──────────────────────────────────────
                    # M̃ = Q_L^T  m_t  Q_R,  then M̃_{ij} /= λ_i^(L,α) · λ_j^(R,α)
                    M_white = evec_L.T @ buf @ evec_R          # rotate into eigenbasis
                    M_white = M_white / scale_L.unsqueeze(1)   # scale rows by λ_L^(-α)
                    M_white = M_white / scale_R.unsqueeze(0)   # scale cols by λ_R^(-α)
 
                    # ── Step 6: LMO in whitened space ─────────────────────────
                    # u = lmo_norm(M̃)
                    # The whitened gradient is always 2D; Auto dispatches to Spectral.
                    u = norm_backend.lmo(M_white)
 
                    # ── Step 7: Graft reference norm ───────────────────────────
                    # n* = ‖u‖_F  (measured immediately after LMO, before unwhitening)
                    if apply_grafting:
                        graft_norm = u.norm()
 
                    # ── Step 8: Dual norm (duality gap contribution) ───────────
                    # ρ = ⟨lmo(M̃), M̃⟩  computed in whitened space so it is
                    # invariant to the unwhitening distortion.
                    if use_dual_norm:
                        rho = (u * M_white).sum()
                    else:
                        rho = torch.tensor(1.0, device=g.device, dtype=torch.float32)
 
                    # ── Step 9: Unwhitening ────────────────────────────────────
                    # u_{ij} /= λ_i^(L,α) · λ_j^(R,α),  then u = Q_L u Q_R^T
                    u = u / scale_L.unsqueeze(1)
                    u = u / scale_R.unsqueeze(0)
                    u = evec_L @ u @ evec_R.T
 
                    # ── Step 10: Grafting ──────────────────────────────────────
                    # u ← (n* / ‖u‖_F) · u
                    # Restores the Frobenius norm that the eigenvalue round-trip
                    # (whitening × 2) would otherwise distort by λ^(-2α) per side.
                    if apply_grafting:
                        u_norm = u.norm()
                        if u_norm > eps:
                            u = (graft_norm / u_norm) * u
 
                # ── Step 11: Parameter update (both branches) ─────────────────
                # w_{t+1} = (1 - η) w_t - η · s · ρ · u
                update = (scale * rho * u).reshape(g.shape)
 
                if not unconstrained:
                    p.data.mul_(1.0 - lr)          # project toward origin (Frank-Wolfe)
                p.data.add_(update, alpha=-lr)
 
        return loss
 
    # ------------------------------------------------------------------
    # Weight initialisation (delegates to norm classes, same as Scion)
    # ------------------------------------------------------------------
 
    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            scale        = group['scale']
            for p in group['params']:
                norm_backend.init(p)
                p.data *= scale