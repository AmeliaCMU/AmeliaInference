import torch

from shapely import LineString
import numpy as np


def compute_collision(A, B, coll_thresh: float = 0.3):
    # A: 1, T, D
    # B: N, T, D
    breakpoint()

    coll_sum = 0
    seg_a = np.stack([A[:-1], A[1:]], axis=1)
    seg_a = [LineString(x) for x in seg_a]
    for b_sub in B:
        seg_b = np.stack([b_sub[:-1], b_sub[1:]], axis=1)
        seg_b = [LineString(x) for x in seg_b]
        coll = np.linalg.norm(A - b_sub, axis=-1) <= coll_thresh
        coll[1:] |= [x.intersects(y) for x, y in zip(seg_a, seg_b)]
        breakpoint()
        coll_sum += coll.sum()
    return coll_sum


def compute_collisions_to_gt(
    Y_hat: torch.Tensor, Y_gt: torch.Tensor, num_agents: torch.tensor, ego_agent: torch.tensor,
    coll_thresh: float = 0.3
) -> torch.tensor:
    """ Comptues collisions between predicted agent and other agents' ground truth.

    Inputs
    ------
        Y_hat[torch.tensor(B, 1, T, M, D)]: ego agent's prediction.
        Y_gt[torch.tensor(B, A, T, D)]: ground truth scene.
        num_agents[torch.tensor(B)]: number of agents in each scene.
        ego_agent[torch.tensor(B)]: ID of ego agent within the scene.

    Output
    ------
        collisions[torch.tensor(B)]: worst mode's (max. number of) collisions.

    """
    B, A, T, D = Y_gt.shape
    Y_hat = Y_hat[..., -T:, :, :].cpu().numpy()
    Y_gt = Y_gt.cpu().numpy()
    _, _, _, M, _ = Y_hat.shape

    collisions = torch.zeros(size=(B,))

    # breakpoint()
    # Iterating over all scenes
    # TODO: add weigh by Y_hat_scores
    for b in range(B):
        # Iterating over all ego-agent modes
        ego_modes = Y_hat[b, 0]                        # T, M, D
        mask = np.zeros(shape=(A,), dtype=bool)        # A
        mask[:num_agents[b]] = True
        mask[ego_agent[b]] = False
        other_Y = Y_gt[b, mask]                        # A-1, T, D
        collisions[b] = max(
            [compute_collision(ego_modes[..., m, :], other_Y, coll_thresh) for m in range(M)])
    return collisions.to('cuda:0')


def compute_collisions_to_pred(
    Y_hat: torch.Tensor, Y_gt: torch.Tensor, num_agents: torch.tensor, ego_agent: torch.tensor,
    coll_thresh: float = 0.3
) -> torch.tensor:
    """ Comptues collisions amongst scene predictions. Assumes one predicted scene per mode.

    Inputs
    ------
        Y_hat[torch.tensor(B, A, T, M, D)]: scene predictions.
        Y_gt[torch.tensor(B, A, T, D)]: ground truth scene.
        num_agents[torch.tensor(B)]: number of agents in each scene.
        ego_agent[torch.tensor(B)]: ID of ego agent within the scene.

    Output
    ------
        collisions[torch.tensor(B)]: worst mode's (max. number of) collisions.

    """
    B, A, T, D = Y_gt.shape
    Y_hat = Y_hat[..., -T:, :, :].cpu().numpy()
    _, _, _, M, _ = Y_hat.shape

    collisions = torch.zeros(size=(B,))

    # Iterating over all scenes
    for b in range(B):
        ego_modes = Y_hat[b, ego_agent[b]]             # T, M, D
        mask = np.zeros(shape=(A,), dtype=bool)        # A
        mask[:num_agents[b]] = True
        mask[ego_agent[b]] = False
        other_Y = Y_hat[b, mask]                       # A-1, T, M, D
        collisions[b] = max(
            [compute_collision(ego_modes[..., m, :], other_Y[..., m, :], coll_thresh) for m in range(M)])
    return collisions.to('cuda:0')


def compute_collisions_gt2gt(
    Y_hat: torch.Tensor, Y_gt: torch.Tensor, num_agents: torch.tensor, ego_agent: torch.tensor,
    coll_thresh: float = 0.3
) -> torch.tensor:
    """ Comptues collisions amongst ground truth.

        Inputs
        ------
            Y_hat[torch.tensor(B, 1, T, D)]: ego agent's ground truth
            Y_gt[torch.tensor(B, A, T, D)]: ground truth scene.
            num_agents[torch.tensor(B)]: number of agents in each scene.
            ego_agent[torch.tensor(B)]: ID of ego agent within the scene.

        Output
        ------
            collisions[torch.tensor(B)]: worst mode's (max. number of) collisions.

        """

    B, A, T, D = Y_gt.shape
    Y_hat = Y_hat[..., -T:, :, :].cpu().numpy()
    Y_gt = Y_gt.cpu().numpy()

    collisions = torch.zeros(size=(B,))

    for b in range(B):
        ego_gt = Y_hat[b, 0]                          # 1, T, D
        mask = np.zeros(shape=(A,), dtype=bool)        # A
        mask[:num_agents[b]] = True
        mask[ego_agent[b]] = False
        other_Y = Y_gt[b, mask]                        # A-1, T, D
        collisions[b] = max(
            [compute_collision(ego_gt, other_Y, coll_thresh)])
        return collisions.to('cuda:0')
