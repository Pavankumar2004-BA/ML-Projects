import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score



def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    Train_Loss =[]
    Train_Accuracy =[]
    Valid_Loss = []
    Valid_Accuracy=[]
    f1_scores=[]
    for epoch in range(1, args.epochs+1):
        corrects, avg_loss = 0, 0
        for batch in train_iter:
            model.train()
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            # print(feature.size())
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        
        size = len(train_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects/size
        print('\nEpoch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,avg_loss, accuracy,corrects, size))
        Train_Accuracy.append(accuracy.item())
        Train_Loss.append(avg_loss)
        valid_loss,valid_acc,f1_score = eval(dev_iter, model, args)
        Valid_Accuracy.append(valid_acc.item())
        Valid_Loss.append(valid_loss)
        f1_scores.append(f1_score)
        save(model, args.save_dir, 'model', epoch)
    return Train_Accuracy,Train_Loss,Valid_Accuracy,Valid_Loss,f1_scores


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    x=np.empty(1,dtype=int)
    y=np.empty(1,dtype=int)
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        x=np.concatenate((x, torch.max(logit, 1)[1].view(target.size()).data.cpu().numpy()), axis=None)
        y=np.concatenate((y,target.data.cpu().numpy()))
    f1_scr = f1_score(x[1:],y[1:],average='micro')
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('Evaluation - loss: {:.6f} f1-score : {:.4f}  acc: {:.4f}%({}/{})\n'.format(avg_loss,f1_scr, accuracy, corrects, size))
    return avg_loss,accuracy,f1_scr


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
