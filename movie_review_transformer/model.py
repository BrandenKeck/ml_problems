import torch, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryConfusionMatrix

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
        self.max_len = max_len
        self.d_model = d_model
        self.input_emb = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.lrelu = nn.LeakyReLU()
        self.linear1 = nn.Linear(self.d_model*self.max_len, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 64)
        self.linear4 = nn.Linear(64, 2)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True):

        # Transformer Part
        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to("cuda")
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.input_emb(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)

        # Classification Part
        output = torch.reshape(output, (-1, self.d_model*self.max_len))
        output = self.lrelu(self.linear1(output))
        output = self.lrelu(self.linear2(output))
        output = self.lrelu(self.linear3(output))
        output = self.lrelu(self.linear4(output))
        return F.softmax(output, dim=1)

class Movie_Review_Model():

    # Init Network Class
    def __init__(self,
                 ntoken, max_len,
                 d_model=64, nhead=4, 
                 dim_feedforward=256, num_encoder_layers=6, 
                 dropout=0.3, learning_rate=4.12E-5, 
                 batch_size=64, epochs=20000):
        self.ntoken = ntoken
        self.max_len = max_len
        self.d_model = d_model
        self.model = TransformerModel(ntoken, max_len, d_model, nhead,
                                    dim_feedforward, num_encoder_layers, dropout).to('cuda')
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def train_network(self, train_data):

        ll = nn.CrossEntropyLoss()
        oo = optim.Adam(self.model.parameters(), lr=self.lr)
        trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (text, labels) in enumerate(trainloader):
                labels = torch.reshape(labels, (-1, 2)).float()
                oo.zero_grad()
                outputs = self.model(text)
                loss = ll(outputs, labels)
                loss.backward()
                oo.step()
                running_loss += loss.item()

            print(f'[{epoch + 1}] loss: {running_loss / i}')

        print('Finished Training')

    def test_network(self, test_data):
        self.model.eval()
        result = torch.tensor([[0,0],[0,0]]).to("cuda")
        bcm = BinaryConfusionMatrix()
        bcm.cuda()
        testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        for _, (text, labels) in enumerate(testloader):
            lab = torch.argmax(torch.reshape(labels, (-1, 2)).float(), dim=1)
            pred = torch.argmax(self.model(text), dim=1)
            batch_res = bcm(pred, lab)
            result = result + batch_res
        return result
    
    # Save network to disc
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    # Load network from disc
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
