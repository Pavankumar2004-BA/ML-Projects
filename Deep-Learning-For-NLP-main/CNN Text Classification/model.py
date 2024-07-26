import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNNSentenceClassification(nn.Module):
    
    def __init__(self, args):
        super(CNNSentenceClassification, self).__init__()
        self.args = args
        V = args.vocab_size
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Nk = args.num_kernels
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D , padding_idx=args.PAD_IDX)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Nk, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Nk, C)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Nk, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Nk), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Nk)
        output = self.fc1(x)  # (N, C)
        return output
