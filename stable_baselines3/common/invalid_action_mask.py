from typing import Optional, Dict
from gym import spaces
import torch as th

def get_mask_from_infos(infos: Optional[Dict], action_space: spaces.Space, device: th.device):
    if infos is None:
        return None
    else:
        masks = []
        for info in infos:
            if "invalid_action_mask" in info:
                masks.append(info.get("invalid_action_mask"))
            else:
                masks.append(th.ones(action_space.shape, dtype=th.bool, device=device))
        return th.stack(masks)