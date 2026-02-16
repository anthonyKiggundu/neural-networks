import torch

def rollout_world_model(model, S0, E_seq, horizon):
    """
    Roll the world model forward using its own predictions.

    S0: initial state (B,C,H,W)
    E_seq: event sequence (B, horizon, event_dim)
    """
    states = []
    S_t = S0

    for t in range(horizon):
        E_t = E_seq[:, t]
        S_t = model(S_t, E_t)   # feed prediction back in
        states.append(S_t)

    return torch.stack(states, dim=1)  # (B, T, C, H, W)

