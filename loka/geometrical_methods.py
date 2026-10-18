import torch
from tqdm import tqdm

def herding_order(X, max_ratio, device):
  n = X.shape[0]
  k = int(max_ratio * n) # |S|
  phi = torch.as_tensor(X.reshape(n, -1), dtype=torch.float32, device=device) # phi(x) = x, flattened; float to avoid uint8 overflow
  mu = phi.mean(dim=0)
  m = mu.clone()
  selected = torch.zeros(n, dtype=torch.bool, device=device)
  order = torch.empty(k, dtype=torch.long, device=device)
  for t in tqdm(range(k), desc='herding'):
      scores = (phi @ m).masked_fill(selected, -torch.inf) # remove those that have been selected
      i = torch.argmax(scores)
      selected[i] = True
      order[t] = i
      m += mu - phi[i]
  return order.cpu().numpy()


def k_center_order(X, max_ratio, device, S0=None):
    n = X.shape[0]
    k = int(max_ratio * n)  # |S \ S0|
    phi = torch.as_tensor(X.reshape(n, -1), dtype=torch.float32, device=device)  # flattened; float to avoid uint8 overflow
    sq = phi.pow(2).sum(-1)  # ||x||^2, precomputed once

    def sq_dist_to(C):  # squared distance from every point to its nearest row of C -> [n]
        d = sq[:, None] + C.pow(2).sum(-1) - 2 * phi @ C.T
        return d.clamp_min(0).min(dim=1).values

    if isinstance(S0, str) and S0 == 'rand':
        # Per-dimension min and max across all points -> shape: [1, dim]
        low = phi.min(dim=0, keepdim=True).values
        high = phi.max(dim=0, keepdim=True).values
        
        # Sample uniformly within [low, high] for each dimension independently
        C0 = low + (high - low) * torch.rand(100, phi.shape[1], device=device)
        min_dist = sq_dist_to(C0)
    elif S0 is not None:
        C0 = torch.as_tensor(S0.reshape(S0.shape[0], -1), dtype=torch.float32, device=device)
        min_dist = sq_dist_to(C0)
    else:
        min_dist = torch.full((n,), torch.inf, device=device)

    selected = torch.zeros(n, dtype=torch.bool, device=device)
    order = torch.empty(k, dtype=torch.long, device=device)
    for t in tqdm(range(k), desc='k-center'):
        scores = min_dist.masked_fill(selected, -torch.inf)  # remove those that have been selected
        i = torch.argmax(scores)
        selected[i] = True
        order[t] = i
        min_dist = torch.minimum(min_dist, sq_dist_to(phi[i:i+1]))  # only the new center can shrink distances
    return order.cpu().numpy()