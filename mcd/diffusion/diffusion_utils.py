import torch
from torch.nn import functional as F
import numpy as np
from utils import PlaceHolder


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

def custom_beta_schedule_discrete(timesteps, average_num_nodes=30, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()



def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_discrete_features(probX, probE, node_mask, step=None, add_nose=True):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = probX.shape

    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    probX = probX + 1e-12
    probX = probX / probX.sum(dim=-1, keepdim=True)
    X_t = probX.multinomial(1)      # (bs * n, 1)
    X_t = X_t.reshape(bs, n)        # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)           # (bs * n * n, de_out)
    probE = probE + 1e-12
    probE = probE / probE.sum(dim=-1, keepdim=True)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)    # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    X_t = X_t.float()
    Qt_T = Qt.transpose(-1, -2).float()                                       # bs, N, dt
    assert Qt.dim() == 3
    left_term = X_t @ Qt_T
    left_term = left_term.unsqueeze(dim=2)                    # bs, N, 1, d_t-1
    right_term = Qsb.unsqueeze(1) 
    numerator = left_term * right_term                        # bs, N, d0, d_t-1
    
    denominator = Qtb @ X_t.transpose(-1, -2)                 # bs, d0, N
    denominator = denominator.transpose(-1, -2)               # bs, N, d0
    denominator = denominator.unsqueeze(-1)                   # bs, N, d0, 1

    denominator[denominator == 0] = 1.
    return numerator / denominator


def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    # Add a small value everywhere to avoid nans
    pred_X = pred_X.clamp_min(1e-18)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)

    pred_E = pred_E.clamp_min(1e-18)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = torch.ones(true_X.size(-1), dtype=true_X.dtype, device=true_X.device)
    row_E = torch.zeros(true_E.size(-1), dtype=true_E.dtype, device=true_E.device).clamp_min(1e-18)
    row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    true_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_X[~node_mask] = row_X
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    return true_X, true_E, pred_X, pred_E

def posterior_distributions(X, X_t, Qt, Qsb, Qtb, X_dim):
    bs, n, d = X.shape
    X = X.float()
    Qt_X_T = torch.transpose(Qt.X, -2, -1).float()                  # (bs, d, d)
    left_term = X_t @ Qt_X_T                                        # (bs, N, d)
    right_term = X @ Qsb.X                                          # (bs, N, d)
    
    numerator = left_term * right_term                              # (bs, N, d)
    denominator = X @ Qtb.X                                         # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denominator = denominator * X_t 
    
    num_X = numerator[:, :, :X_dim]
    num_E = numerator[:, :, X_dim:].reshape(bs, n*n, -1)

    deno_X = denominator[:, :, :X_dim]
    deno_E = denominator[:, :, X_dim:].reshape(bs, n*n, -1)

    # denominator = (denominator * X_t).sum(dim=-1)                   # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    denominator = denominator.unsqueeze(-1)                         # (bs, N, 1)

    deno_X = deno_X.sum(dim=-1).unsqueeze(-1)
    deno_E = deno_E.sum(dim=-1).unsqueeze(-1)

    deno_X[deno_X == 0.] = 1
    deno_E[deno_E == 0.] = 1
    prob_X = num_X / deno_X
    prob_E = num_E / deno_E
    
    prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True)
    prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True)
    return PlaceHolder(X=prob_X, E=prob_E, y=None)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """ Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_X = F.one_hot(U_X.long(), num_classes=x_limit.shape[-1]).float()

    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_E = F.one_hot(U_E.long(), num_classes=e_limit.shape[-1]).float()

    U_X = U_X.to(node_mask.device)
    U_E = U_E.to(node_mask.device)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = (U_E + torch.transpose(U_E, 1, 2))

    assert (U_E == torch.transpose(U_E, 1, 2)).all()
    return PlaceHolder(X=U_X, E=U_E, y=None).mask(node_mask)

def index_QE(X, q_e, n_bond=5):
    bs, n, n_atom = X.shape
    node_indices = X.argmax(-1)  # (bs, n)

    exp_ind1 = node_indices[ :, :, None, None, None].expand(bs, n, n_atom, n_bond, n_bond)
    exp_ind2 = node_indices[ :, :, None, None, None].expand(bs, n, n, n_bond, n_bond)
    
    q_e = torch.gather(q_e, 1, exp_ind1)
    q_e = torch.gather(q_e, 2, exp_ind2) # (bs, n, n, n_bond, n_bond)


    node_mask = X.sum(-1) != 0
    no_edge = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
    q_e[no_edge] = torch.tensor([1, 0, 0, 0, 0]).type_as(q_e)

    return q_e
