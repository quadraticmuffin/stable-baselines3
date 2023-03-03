from typing import Optional, Dict
from gym import spaces
import torch as th

def get_mask_from_infos(infos: Optional[Dict], action_space: spaces.Space, device: th.device):
    if infos is None:
        return None
    else:
        return th.stack([
            th.Tensor(
                info.get("invalid_action_mask", th.ones(action_space.shape))
            ) for info in infos
        ]).to(dtype=th.bool, device=device)