import torch, torchrl
import numpy as np
from tensordict import TensorDict


from problem import LeftRight
lr = LeftRight(50, 0, -100, 100)

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

