from model import MNIST_Model

mod = MNIST_Model()
mod.load("mod.pt")
mod.test_network()
