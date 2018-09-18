from random import seed, shuffle
import numpy as np
from io import TextIOWrapper
import os
import sys
import pickle
from datetime import datetime
from socket import gethostname
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from bow import vocab, train, valid, test

sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

seed(123)
np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    print('# cuda')
    torch.cuda.manual_seed(123)

z_dim = 100
X_dim = len(vocab)

eps_dim = 300
h_dim = 800
#h_dim = int(sys.argv[1])

mb_size = 200
q_lr = 0.0001
d_lr = 0.0001
p_lr = 0.003
myReLU = torch.nn.LeakyReLU
weight_init = torch.nn.init.xavier_normal_

setting_str = '_e{:d}_h{:d}_b{:d}_q{}_d{}_p{}_'.format(
    eps_dim, h_dim, mb_size, q_lr, d_lr, p_lr)
print('# setting: {}'.format(setting_str))

def log(x):
    return torch.log(x + 1e-10)

# Encoder: f_Q(z|x,eps)
Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim + eps_dim, h_dim * 2),
    myReLU(),
    torch.nn.Linear(h_dim * 2, z_dim)
)

for l in Q:
    if type(l) == myReLU:
        weight_init(prev_l.weight)
    prev_l = l

# Discriminator: f_D(X, z)
D = torch.nn.Sequential(
    torch.nn.Linear(X_dim + z_dim, h_dim * 2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(h_dim * 2, 1)
)

for l in D:
    if type(l) == myReLU:
        weight_init(prev_l.weight)
    prev_l = l

# Decoder: f_P(phi|z)
P = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim * 2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(h_dim * 2, X_dim)
)

for l in P:
    if type(l) == myReLU:
        weight_init(prev_l.weight)
    prev_l = l

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    Q.cuda()
    D.cuda()
    P.cuda()

def reset_grad():
    Q.zero_grad()
    D.zero_grad()
    P.zero_grad()

def docword2vec(docword):
    vec = np.zeros(X_dim)
    for i in docword:
        vec[i] = docword[i]
    return vec
    
def sample_X(size):
    if 'cnt' not in sample_X.__dict__:
        sample_X.cnt = 0
    X = list()
    doc_cnt = 0
    while doc_cnt < size:
        X.append(docword2vec(train[sample_X.cnt]))
        doc_cnt += 1
        sample_X.cnt += 1
        if sample_X.cnt == len(train):
            shuffle(train)
            sample_X.cnt = 0
    return Variable(torch.FloatTensor(np.array(X))).type(dtype)

def adjust_learning_rate(lr, tau, optimizer, cur_epoch):
    new_lr = lr * ((1 + cur_epoch / 2000.) ** (- tau))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

Q_solver = optim.Adam(Q.parameters(), lr=q_lr)
D_solver = optim.SGD(D.parameters(), lr=d_lr, momentum=0.9)
P_solver = optim.Adam(P.parameters(), lr=p_lr)

dropout = torch.nn.Dropout(p=0.5)

it_log = 10
it_test = 100
it_lr = 1000
it_save = 5000

for it in range(100000):

    X = dropout(sample_X(mb_size))

    # Discriminator
    for _ in range(1):
        eps = Variable(torch.randn(mb_size, eps_dim)).type(dtype)
        z = Variable(torch.randn(mb_size, z_dim)).type(dtype)
        z_sample = Q(torch.cat([X, eps], 1))
        D_q = F.sigmoid(D(torch.cat([X, z_sample], 1)))
        D_prior = F.sigmoid(D(torch.cat([X, z], 1)))

        D_loss = - torch.mean(log(D_q) + log(1.0 - D_prior))
        
        D_loss.backward()
        D_solver.step()
        reset_grad()

    # Decoder/Encoder
    eps = Variable(torch.randn(mb_size, eps_dim)).type(dtype)
    z_sample = Q(torch.cat([X, eps], 1))
    D_sample = D(torch.cat([X, z_sample], 1))
    log_prob = F.log_softmax(P(z_sample), dim=1)

    disc = torch.mean(- D_sample)
    loglike = torch.mean(torch.sum(X * log_prob, dim=1))

    elbo = - (disc + loglike)

    elbo.backward()
    Q_solver.step()
    P_solver.step()
    reset_grad()

    adjust_learning_rate(q_lr, 0.7, Q_solver, it)
    adjust_learning_rate(d_lr, 0.7, D_solver, it)
    adjust_learning_rate(p_lr, 0.7, P_solver, it)
    
    if it % it_log == 0:
        print('iter: {} ELBO: {:.3f} D_loss: {:.3f}'
              .format(it, - elbo.item(), - D_loss.item()), end=' ')
        X = list()
        for docword in valid:
            X.append(docword2vec(docword))
        X = Variable(torch.FloatTensor(np.array(X))).type(dtype)
        eps = Variable(torch.randn(len(valid), eps_dim)).type(dtype)
        z_sample = Q(torch.cat([X, eps], 1))
        log_prob = F.log_softmax(P(z_sample), dim=1)
        print('valid_perp: {:.2f}'
              .format(1.0 / torch.exp(torch.sum(X * log_prob) / torch.sum(X)).item()))

    if it and it % it_lr == 0:
        #q_lr *= 0.1
        #d_lr *= 0.1
        p_lr *= 0.1

    if it and it % it_test == 0:
        X = list()
        for docword in test:
            X.append(docword2vec(docword))
        X = Variable(torch.FloatTensor(np.array(X))).type(dtype)
        perp = list()
        for _ in range(10):
            eps = Variable(torch.randn(len(test), eps_dim)).type(dtype)
            z_sample = Q(torch.cat([X, eps], 1))
            log_prob = F.log_softmax(P(z_sample), dim=1)
            perp.append(1.0 / torch.exp(torch.sum(X * log_prob) / torch.sum(X)).item())
        perp = np.array(perp)
        print('# test_perp: {:.2f} {:.2f}'.format(perp.mean(), perp.std()))

    if it and it % it_save == 0:
        pickle_file = 'param.avbdm.' + setting_str + '.' + gethostname() + '.'
        pickle_file += datetime.now().strftime('%Y%m%d%H') + '.pkl'
        pickle.dump([Q.cpu().state_dict(), D.cpu().state_dict(), P.cpu().state_dict()],
                    open(pickle_file, 'wb'))
        Q.cuda()
        D.cuda()
        P.cuda()
        
    sys.stdout.flush()
