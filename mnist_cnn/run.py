from model import MNIST_Model
from torchvision import datasets, transforms

mod = MNIST_Model()
mod.load("mod.pt")
mod.test_network()
