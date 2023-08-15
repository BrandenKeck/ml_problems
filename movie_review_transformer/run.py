from model import Movie_Review_Model

mod = Movie_Review_Model(120000, 3000)
mod.load("mod.pt")
mod.test_network()
