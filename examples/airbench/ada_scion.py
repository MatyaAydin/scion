import math
import torch


#######################################################
# AdaScion
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
        normalized (bool, optional): If True, normalizes by the input dimension. Use True only for non-input layers.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=False, transpose=False):
        self.normalized = normalized
        self.transpose = transpose
        self.norm_type = "euclidean"

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
        normalized (bool, optional): If True, normalizes by the input dimension. Use False only for the input layer.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=True, transpose=False):
        self.normalized = normalized
        self.transpose = transpose
        self.norm_type = "euclidean"

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
    def __init__(self):
        self.norm_type = "euclidean"

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
        self.norm_type = "spectral"

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
        
        if w.ndim == 3:     # Conv1d
            out_channels, in_channels, k = w_fp.shape
            w_fp.mul_((out_channels / in_channels)**0.5 / k)
        elif w.ndim == 4:     # Conv2d
            out_channels, in_channels, k, _ = w_fp.shape
            w_fp.mul_((out_channels / in_channels)**0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized
        self.norm_type = "spectral"

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        g *= scale

        return g

    def init(self, w):
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp)
        d_out, d_in = w_fp.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        w_fp.mul_(scale)
    
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized
        self.norm_type = "euclidean" # TODO: does it need preconditioning ? euclidean shouldnt change anything but less computation if do nothing

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
            # Generate -1/fan_in or 1/fan_in uniformly at random
            d_out, d_in = w.shape
            w.data = (torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1)
            if self.normalized:
                w.data *= (1/d_in)
        return w


class Auto(Norm):
    def lmo(self, g):
        if g.ndim in [3,4]:
            return SpectralConv().lmo(g)
        elif g.ndim == 2:
            return Spectral().lmo(g)
        elif g.ndim in [0,1]:
            return BiasRMS().lmo(g)

    def init(self, w):
        if w.ndim in [3,4]:
            return SpectralConv().init(w)
        elif w.ndim == 2:
            return Spectral().init(w)
        elif w.ndim in [0,1]:
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


class AdaScion(torch.optim.Optimizer):
    """Scion optimizer implementation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv', 
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)
        scale (float, optional): Scale factor for updates (default: 1.0)
        unconstrained (bool, optional): Whether to use unconstrained updates (default: False)
    
    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'scale': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'scale': radius*60.0,
        ... }]
        >>> optimizer = Scion(optim_groups, lr=2**-12, momentum=0.1)
    """
    def __init__(self, params, lr=1e-3, momentum=1.0, norm: str='Auto', norm_kwargs: dict=None, scale=1.0, unconstrained=False,
                beta_eucl=0.99, beta_spectral=0.999, order=8, eps=1e-8, power_frequency=50, normalize_update=False, use_trace_normalization=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(lr=lr, momentum=momentum, scale=scale, unconstrained=unconstrained, norm=norm, norm_kwargs=norm_kwargs,
        beta_eucl=beta_eucl, beta_spectral=beta_spectral, order=order, eps=eps, power_frequency=power_frequency, normalize_update=normalize_update, use_trace_normalization=use_trace_normalization)
        super().__init__(params, defaults)
        self.effective_lrs = {}

    def step(self):
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            order = group["order"]
            beta_eucl = group["beta_eucl"]
            beta_spectral = group["beta_spectral"]
            eps = group["eps"]
            power_frequency = group["power_frequency"]
            normalize_update = group["normalize_update"]
            use_trace_normalization = group["use_trace_normalization"]

            norm_type = norm_backend.norm_type

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                g_2d = to_2d(g).float()

                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                iter_number = state["step"]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.clone(g_2d)
                buf = state['momentum_buffer']
                if iter_number > 1:
                    buf.mul_(momentum).add_(g_2d, alpha=1. - momentum)


                if norm_type == "euclidean":

                    if "euclidean_buffer" not in state.keys():
                        state["euclidean_buffer"] =  torch.zeros_like(g_2d)

                    eucl_buffer = state["euclidean_buffer"]
                    eucl_buffer.mul_(beta_eucl).add_(g_2d * g_2d, alpha=1 - beta_eucl)

                    bias_correction_M = 1. - momentum**iter_number
                    bias_correction_P = 1. - beta_eucl**iter_number

                    m_hat = buf / bias_correction_M
                    p_hat = eucl_buffer.sqrt() / bias_correction_P**0.5

                    lmo_ = weighted_norm_D_lmo(norm_backend, m_hat, p_hat, eps)
                    dual_norm = (lmo_ * m_hat).sum()
                    if normalize_update:
                        dual_norm /= min(g_2d.shape[0], g_2d.shape[1])
                    update = scale * lmo_ * dual_norm
                    effective_lr = scale * dual_norm * lr
                    self.effective_lrs[group_idx] = effective_lr

                else: # spectral

                    if "L_buffer" not in state.keys():
                        state["L_buffer"] = eps * torch.eye(g_2d.shape[0], g_2d.shape[0], device=g.device, dtype=g_2d.dtype)
                        state["R_buffer"] = eps * torch.eye(g_2d.shape[1], g_2d.shape[1], device=g.device, dtype=g_2d.dtype)
                        state["L_inv"] = None
                        state["R_inv"] = None

                    L = state["L_buffer"]
                    R = state["R_buffer"]

                    L.mul_(beta_spectral).add_(g_2d.matmul(g_2d.T), alpha=1 - beta_spectral)
                    R.mul_(beta_spectral).add_(g_2d.T.matmul(g_2d), alpha=1 - beta_spectral)
                    # L.add_(g_2d.matmul(g_2d.T), alpha=1.)
                    # R.add_(g_2d.T.matmul(g_2d), alpha=1.)

                    if iter_number % power_frequency == 1 or state["L_inv"] is None:
                        L_inv, R_inv = compute_LR_inv(L, R, order=order, eps_stab=eps, use_trace_normalization=use_trace_normalization)
                        state["L_inv"] = L_inv
                        state["R_inv"] = R_inv

                    L_inv = state["L_inv"]
                    R_inv = state["R_inv"]

                    lmo_ = weighted_norm_LR_lmo(norm_backend, buf, L_inv, R_inv)
                    dual_norm = (lmo_ * buf).sum()
                    if normalize_update:
                        dual_norm /= min(g_2d.shape[0], g_2d.shape[1])
                    update = scale * lmo_ * dual_norm
                    effective_lr = scale * dual_norm * lr
                    self.effective_lrs[group_idx] = effective_lr



                update = update.reshape(g.shape)

                if not unconstrained:
                    p.data.mul_(1-lr)
                p.data.add_(update, alpha=-lr)


    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            init_func = norm_backend.init
            scale = group['scale']
            for p in group['params']:
                init_func(p)
                p.data *= scale


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    From: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X.float()



def matrix_power(matrix, power):
    """
    From https://github.com/moskomule/shampoo.pytorch/blob/master/shampoo.py
    """

    # matrix = matrix.cpu()
    # matrix = matrix.float()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t())

def to_2d(g):
    """Reshape gradient to 2D: [first_dim, rest_flattened]"""
    if g.dim() == 1:
        return g.unsqueeze(0)   # [1, d] — treat as single row
    return g.reshape(g.shape[0], -1)  # [d0, d1*d2*...] for conv/linear weights

def weighted_norm_D_lmo(norm, G, D, eps=1e-4):
    D_inv = 1. / (D + eps)
    return D_inv * norm.lmo(D_inv * G)

def compute_LR_inv(L, R, order=8, eps_stab=1e-4, use_trace_normalization=False):
    p = 1.0 / order 

    dim_L = L.size(0)
    dim_R = R.size(0)

    if use_trace_normalization:
        trace_L = torch.trace(L)
        trace_R = torch.trace(R)

        L_tilde = L * (dim_L / (trace_L + eps_stab))
        R_tilde = R * (dim_R / (trace_R + eps_stab))
    else:
        L_tilde = L
        R_tilde = R

    L_d = L_tilde.double() + eps_stab * torch.eye(dim_L, device=L.device, dtype=torch.float64)
    R_d = R_tilde.double() + eps_stab * torch.eye(dim_R, device=R.device, dtype=torch.float64)

    try:
        sigma_L, U_L = torch.linalg.eigh(L_d)
        sigma_R, U_R = torch.linalg.eigh(R_d)
    except torch._C._LinAlgError:
        print('CPU fallback')
        sigma_L, U_L = torch.linalg.eigh(L_d.cpu())
        sigma_R, U_R = torch.linalg.eigh(R_d.cpu())
        sigma_L, U_L = sigma_L.to(L.device), U_L.to(L.device)
        sigma_R, U_R = sigma_R.to(R.device), U_R.to(R.device)

    sigma_L = torch.clamp(sigma_L.float(), min=eps_stab)
    sigma_R = torch.clamp(sigma_R.float(), min=eps_stab)
    U_L = U_L.float()
    U_R = U_R.float()

    sigma_L_inv = sigma_L ** (-p)
    sigma_R_inv = sigma_R ** (-p)

    L_inv = (U_L @ torch.diag(sigma_L_inv) @ U_L.T).to(L.dtype)
    R_inv = (U_R @ torch.diag(sigma_R_inv) @ U_R.T).to(R.dtype)
    
    return L_inv, R_inv

def weighted_norm_LR_lmo(norm, G, L_inv, R_inv):
    precond_G = L_inv.matmul(G).matmul(R_inv)
    return L_inv.matmul(norm.lmo(precond_G)).matmul(R_inv)



