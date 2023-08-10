import torch, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from data import train_data, test_data

class PositionalEncoding(nn.Module):

    """
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    """
    Args:
        x: [sequence length, batch size, embed dim]
        output: [sequence length, batch size, embed dim]
    """
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Transformer):

    """
    Args:
    ntoken: size of the dictionary of embeddings
        (i.e.) the length of the vocabulary in the dictionary
    d_model: the number of expected features in the encoder/decoder inputs.
        (or) the size of each embedding vector.
    nhead: the number of heads in the multiheadattention models.
    dim_feedforward: the dimension of the feedforward network model.
    num_encoder_layers: the number of sub-encoder-layers in the encoder.
    dropout: the dropout value.
    max_len: the maximum length of the incoming sequence.
    """
    def __init__(self, 
                 ntoken, max_len,
                 d_model, nhead, 
                 dim_feedforward, num_encoder_layers, 
                 dropout):
        super(TransformerModel, self).__init__(d_model=d_model, 
                                               nhead=nhead, 
                                               dim_feedforward=dim_feedforward, 
                                               num_encoder_layers=num_encoder_layers)
        self.src_mask = None
        self.d_model = d_model
        self.input_emb = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.input_emb(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

class Movie_Review_Model():

    # Init Network Class
    def __init__(self,
                 ntoken, max_len,
                 d_model=128, nhead=8, 
                 dim_feedforward=2048, num_encoder_layers=6, 
                 dropout=0.2, learning_rate=4.12E-3, 
                 batch_size=64, epochs=2000):
        self.model = TransformerModel(ntoken, max_len, d_model, nhead,
                                    dim_feedforward, num_encoder_layers, dropout)
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def train_network(self):

        ll = nn.NLLLoss()
        oo = optim.Adam(self.model.parameters(), lr=self.lr) 
        trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (text, labels) in enumerate(trainloader):
                oo.zero_grad()
                outputs = self.model(text)
                outputs = outputs.view(-1, self.model)
                loss = ll(outputs, labels)
                loss.backward()
                oo.step()
                running_loss += loss.item()

            print(f'[{epoch + 1}] loss: {running_loss / i}')

        print('Finished Training')

    def test_network(self, ):
        testloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)
        text, labels = next(iter(testloader))
        res = torch.argmax(self.model(text), dim=1)
        print(f"Sample Results: {res[:10]}")
        print(f"Sample Labels: {labels[:10]}")
        print(f"Accuracy: {torch.sum(torch.eq(res, labels))/len(valset)}")
    
    # Save network to disc
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    # Load network from disc
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
