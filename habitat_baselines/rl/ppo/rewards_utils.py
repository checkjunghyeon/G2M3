import numpy as np
import torch


def compute_semantic_map_acc(mapCache, grid_map):
    # Perform channel-wise labeling to represent the label with the highest probability.
    grid_label_map = torch.argmax(grid_map, dim=1, keepdim=True)  # B x C x cH x cW

    # Get the GT goal map.
    mapCache = mapCache.squeeze()
    nonzero_mask = grid_label_map != 0  # B x C x cH x cW

    # Use a comparison operation to find the matching location.
    matches = ((grid_label_map == mapCache.unsqueeze(1)) & nonzero_mask).sum(dim=(2, 3))  # B x C

    # Save the number of matches in each batch as a list.
    counts = matches.sum(dim=1).tolist()
    rewards = compute_accuracy_reward(np.array(counts))

    return rewards


def compute_goal_map_acc(observations, grid_map):
    # Perform channel-wise labeling to represent the label with the highest probability.
    grid_label_map = torch.argmax(grid_map, dim=1, keepdim=True)  # B x C x cH x cW

    # Get the GT goal map.
    mapCache = observations["obsMap"][:, :, :, 1]
    nonzero_mask = grid_label_map != 0  # B x C x cH x cW

    # Use a comparison operation to find the matching location.
    matches = ((grid_label_map == mapCache.unsqueeze(1)) & nonzero_mask).sum(dim=(2, 3))  # B x C

    # Save the number of matches in each batch as a list.
    counts = matches.sum(dim=1, dtype=np.float).tolist()
    rewards = compute_accuracy_reward(np.array(counts))

    return rewards


def compute_accuracy_reward(counts, reward_scale=0.005):
    rewards = (counts * reward_scale).tolist()

    return rewards
