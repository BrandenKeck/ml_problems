#from data import *
#len(rdict.word2idx)
# -> >100,000 Words
# -> Pad to 3000 Length

from model import Movie_Review_Model

mod = Movie_Review_Model(120000, 
                         3000, 
                         d_model=16,
                         nhead=4,
                         dim_feedforward=128,
                         num_encoder_layers=3,
                         batch_size=8)
mod.train_network()
mod.save("mod.pt")

# # TEST
# loader = DataLoader(train_data, batch_size=2, shuffle=True) 
# for i, (text, labels) in enumerate(loader): src=text
# src

# mod.model._generate_square_subsequent_mask(3000)
# src = mod.model.input_emb(src) * math.sqrt(mod.model.d_model)
# mod.model.pos_encoder(src)
# mod.model.encoder(src, mask=mod.model.src_mask)
# mod.model.decoder(src)

# n, d, m = 3, 5, 7
# embedding = nn.Embedding(n, d)
# idx = torch.tensor([[1, 2],[1, 2]])
# embedding(idx).shape

# idx.shape
# src.shape
# src2 = 29000*torch.ones(2, 3000, dtype=torch.int32)
# etest = nn.Embedding(90000, 128)
# etest(src2).shape

# torch.max(src)
