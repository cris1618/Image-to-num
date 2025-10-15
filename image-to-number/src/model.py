import torch
import torch.nn as nn
import torch.nn.functional as F

# define the class for the CRNN
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        """
        imgH: height of the input image
        nc: number of channels (1 for grayscale, 3 for RGB)
        nclass: number of output classes (digits 0-9, plus blank for CTC)
        nh: size of the LSTM hidden state
        """
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        # Calculate feature dimensions after CNN
        self.rnn_input_size = 128 * (imgH // 4)

        # Define RNN layers
        self.rnn = nn.Sequential(
            nn.LSTM(self.rnn_input_size, nh, bidirectional=True, batch_first=True),
            # The output layer that maps hidden state to classes
            nn.Linear(nh * 2, nclass)
        )

    def forward(self, x):
        conv_out = self.cnn(x)
        batch_size, channels, height, width = conv_out.size()

        # Reshape for RNN: collapse height dimension into channels
        conv_out = conv_out.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        conv_out = conv_out.contiguous().view(batch_size, width, -1)  # (batch_size, width, rnn_input_size)

        # Run through RNN; output shape: (batch_size, seq_len, nclass)
        rnn_out, _ = self.rnn[0](conv_out)
        output = self.rnn[1](rnn_out)
        
        return output

def get_model():
    imgH = 32
    nc = 1
    nclass = 11 # Digits 0-9 and one blank for CTC
    nh = 256

    model = CRNN(imgH, nc, nclass, nh)

    return model





