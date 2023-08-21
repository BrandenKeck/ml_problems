import os
from data import get_dataset
from model import Movie_Review_Model

DEVICE='cuda:1'

mod = Movie_Review_Model(120000, 3000, batch_size=12, device=DEVICE)
if os.path.isfile("mod.pt"): 
    mod.load("mod.pt")
test_data = get_dataset(10000, "test", device=DEVICE)
res = mod.test_network(test_data)
print(res)
