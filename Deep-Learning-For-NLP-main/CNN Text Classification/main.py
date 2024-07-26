#! /usr/bin/env python
import os
import argparse
import datetime
import random
import numpy as np
import torch
from torchtext.vocab import GloVe, FastText
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchtext.legacy.data import Field, BucketIterator, Iterator
from torchtext.legacy import data
import model
import train
import mydatasets
from torchtext.vocab import GloVe, FastText

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=25, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=50, help='batch size for training [default: 64]')
parser.add_argument('-save-dir', type=str, default='models', help='where to save the snapshot')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-num-kernels', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-glove',type=bool , default=False , help ='fix glove embeddins')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=4, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-model', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


#No of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter

def assign0(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.Assign0.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data , vectors = "glove.840B.300d",unk_init = torch.Tensor.normal_)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter

def assign1(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.Assign1.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data , vectors = "glove.840B.300d",unk_init = torch.Tensor.normal_)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter

def assign2(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.Assign2.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data , vectors = "glove.840B.300d",unk_init = torch.Tensor.normal_)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter

# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
# train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter = assign0(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter = assign1(text_field, label_field, device=-1, repeat=False)
train_iter, dev_iter = assign2(text_field, label_field, device=-1, repeat=False)



# update args and print
args.vocab_size = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
print(args.cuda)
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# args.save_dir = os.path.join(args.save_dir, "Dataset0")
# args.save_dir = os.path.join(args.save_dir, "Dataset1")
args.save_dir = os.path.join(args.save_dir, "Dataset2")

PAD_IDX = text_field.vocab.stoi[text_field.pad_token]
args.PAD_IDX = PAD_IDX


#Parameters
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


#Random Intialize
def set_seed(seed = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(1)

# model
cnn = model.CNNSentenceClassification(args)
pretrained_embeddings = text_field.vocab.vectors
cnn.embed.weight.data.copy_(pretrained_embeddings)

print(f'The model has {count_parameters(cnn):,} trainable parameters')

if args.model is not None:
    print('\nLoading model from {}...'.format(args.model))
    cnn.load_state_dict(torch.load(args.model))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
        

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train_accuracy,train_loss,valid_accuracy,valid_loss,f1_scores=train.train(train_iter, dev_iter, cnn, args)
        ma=0
        ind=0
        for i in range(args.epochs):
            if(f1_scores[i]>ma):
                ma = f1_scores[i]
                ind =i
        print(ma,ind)
        text = open("CNN_Dataset2_Results","w")
        text.write("Train Accuracy : "+str(train_accuracy)+"\n")
        text.write("Train Loss : "+str(train_loss)+"\n")
        text.write("Valid Accuracy : "+str(valid_accuracy)+"\n")
        text.write("Valid Loss : "+str(valid_loss)+"\n")
        text.write("F1-Score : "+str(f1_scores)+"\n")
        text.write(str(ma)+" "+str(ind))
        text.close()
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

