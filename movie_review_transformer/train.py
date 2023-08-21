import os
from data import get_dataset
from model import Movie_Review_Model

DEVICE='cuda:0'

super_epochs = 100
mod = Movie_Review_Model(120000, 3000,
                         batch_size=12, learning_rate=4.12E-5,
                         epochs=10, device=DEVICE)
if os.path.isfile("mod.pt"): 
    mod.load("mod.pt")
for _ in range(super_epochs):
    train_data = get_dataset(5000, "train", device=DEVICE)
    mod.train_network(train_data)
    mod.save("mod.pt")
