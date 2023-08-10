from model import Movie_Review_Model

mod = Movie_Review_Model()
mod.load("mod.pt")
mod.test_network()
