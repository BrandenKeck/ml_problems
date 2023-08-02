from model import MNIST_Model

mod = MNIST_Model()
mod.train_network()
mod.save("mod.pt")
