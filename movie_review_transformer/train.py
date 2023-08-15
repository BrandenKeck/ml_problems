import os
from model import Movie_Review_Model
from data import get_dataset

super_epochs = 40
mod = Movie_Review_Model(120000, 3000,
                         batch_size=12, learning_rate=8.24E-6,
                         epochs=10)
if os.path.isfile("mod.pt"): 
    mod.load("mod.pt")
for _ in range(super_epochs):
    train_data = get_dataset(10000, "train")
    mod.train_network(train_data)
    mod.save("mod.pt")


###
# Playing
###

import torch
from data import rdict

# 0
xx = "Heart of Stone joins the streamer’s seemingly endless stable of star-driven action films, but it does little to differentiate itself... There’s no big action sequence, or even a single moment that merits any sort of feeling other than déjà vu."
xxt = torch.tensor([rdict.tokenize(xx)]).to("cuda")
mod.model(xxt)

# 1
xx = "This is one of the stronger recent Netflix original films to date, at least from the subgenre one might entitle Action Dramas Featuring an A-List Star That Won’t Be Getting an Oscar Push."
xxt = torch.tensor([rdict.tokenize(xx)]).to("cuda")
mod.model(xxt)

# 0
xx = "[An] amusing but instantly forgettable romp. The unstoppable force of Lawrence’s charisma notwithstanding, this is not so much tasteless, just a bit bland."
xxt = torch.tensor([rdict.tokenize(xx)]).to("cuda")
mod.model(xxt)

# 1
xx = "I am about to recommend a mildly raunchy R-rated comedy for one simple reason: I had fun watching it. I am so sick and tired of overlong, overproduced, formulaic “content” that this insignificant movie looks like a work of genius in comparison."
xxt = torch.tensor([rdict.tokenize(xx)]).to("cuda")
mod.model(xxt)