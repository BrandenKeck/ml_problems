import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import ReplayBuffer

# Create a TensorDict
def get_dict(batch_size = 5):
    return (TensorDict(
        source={
            "x": torch.rand(batch_size, 3),
            "y": torch.rand(batch_size, 3),
        },
        batch_size=[batch_size],
    ))

rb = ReplayBuffer(collate_fn=lambda x: x)
rb.extend([get_dict() for _ in np.arange(20)])

print(rb.sample(1))

