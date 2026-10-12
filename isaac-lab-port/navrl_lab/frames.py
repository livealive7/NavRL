"""Goal-frame coordinate helpers (direct ports of NavRL's ``vec_to_new_frame`` / ``vec_to_world``)."""

import torch


def vec_to_new_frame(vec: torch.Tensor, goal_direction: torch.Tensor) -> torch.Tensor:
    """Express ``vec`` (N,3) or (N,K,3) in the frame whose x-axis is ``goal_direction`` (N,3), z ~ world up."""
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_up = torch.tensor([0.0, 0.0, 1.0], device=vec.device).expand_as(x)
    y = torch.cross(z_up, x, dim=-1)
    y = y / y.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y, dim=-1)
    z = z / z.norm(dim=-1, keepdim=True)
    basis = torch.stack([x, y, z], dim=-1)  # (N,3,3), columns are the new axes
    if vec.dim() == 3:
        return torch.bmm(vec, basis)
    return torch.bmm(vec.unsqueeze(1), basis).squeeze(1)


def vec_to_world(vec: torch.Tensor, goal_direction: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`vec_to_new_frame`: goal-frame vector -> world frame."""
    world_x = torch.tensor([1.0, 0.0, 0.0], device=vec.device).expand_as(goal_direction)
    world_in_goal = vec_to_new_frame(world_x, goal_direction)
    return vec_to_new_frame(vec, world_in_goal)
