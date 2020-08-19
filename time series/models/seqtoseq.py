import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
device = 'cpu'


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttenDecoder(nn.Module):
    def __init__(self, input_size, encoder_length, hidden_size, output_size, dropout, num_layers):
        super(AttenDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embeding = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.atten = nn.Linear(hidden_size * 2, encoder_length)
        self.atten_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.output_embeding = nn.Linear(hidden_size, output_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.encoder_length = encoder_length
        self.softmax = nn.Softmax()

    def forward(self, input, hidden, encoder_outputs):
        input = self.embeding(input.unsqueeze(0))
        input = self.dropout(input)
        atten_weight = torch.zeros((input.shape[1], self.encoder_length))
        atten_weight = F.softmax(self.atten(torch.cat((input, hidden),2)),2)
     #   for i in range(input.shape[1]):
     #      atten_weight[i,:] =self.softmax(self.atten(torch.cat((input[0, i], hidden[0,0]))))
        atten_apply = torch.bmm(atten_weight.permute(1,0,2), encoder_outputs.permute(1,0,2))
        atten_combine = self.atten_combine(torch.cat((atten_apply.permute(1,0,2), input),2))
        output, hn = self.gru(atten_combine)
        output = nn.Sigmoid()(self.output_embeding(output))
        return output, hn, atten_weight

    def initHidden(self, batchsize):
        return torch.zeros(1, batchsize, self.hidden_size, device=device)


class SeqtoSeq(nn.Module):
    def __init__(self, inputsize, hiddensize, num_layers, encoder_length, outputsize, dropout,use_teacher):
        # self.encoder = EncoderRNN(inputsize,hiddensize
        super(SeqtoSeq, self).__init__()
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.num_layers = num_layers
        self.outputsize = outputsize
        self.use_teacher = use_teacher
        self.encoder = nn.GRU(inputsize, hiddensize, num_layers)
        self.decoder = AttenDecoder(inputsize, encoder_length, hiddensize, outputsize, dropout, num_layers)

    def forward(self, input, output, output_length):
        encoder_output, encoder_hidden = self.encoder(input.permute(1,0,2))
        decoder_inint_hidden = self.decoder.initHidden(input.shape[0])
        output = torch.zeros((input.shape[0], output_length, self.outputsize),device = device)
        #input = input.unsqeeze(0)
        for i in range(output_length):
            if (i == 0):
                decoder_output, decoder_hidden,atten_weight = self.decoder(input[:, -1, :], decoder_inint_hidden, encoder_output)
            else:
                if(np.random.rand()>self.use_teacher):
                   decoder_output, decoder_hidden,atten_weight = self.decoder(output[:, i - 1, :].squeeze(0), decoder_hidden, encoder_output)
                else:
                   decoder_output, decoder_hidden,atten_weight = self.decoder(decoder_output.squeeze(0), decoder_hidden, encoder_output)
            output[:, i, :] = decoder_output[0]
        return output












