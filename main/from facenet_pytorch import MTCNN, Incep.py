import os
from torch.hub import get_dir

model_dir = os.path.join(get_dir(), 'checkpoints')
print(model_dir)