"""
Module contains functions to transform scores
"""

import torch
from torch import nn
import torch.nn.functional as F

import monotonous_net

BATCH_SIZE = 64
LR = 5e-4

def transform_scores(scores_train, true_false_train, classifier_type='monoton'):
    """
    Transforms already computed scores to give them some form of calibration.
    Arguments:
        - scores_train: scores on the training set
        - true_false_train: list of 0/1 indicating whether the prediction was correct or not
        - classifier_type: 'monoton' or 'unrestricted'
    Returns:
        - scores_train_class: scores on the training set after transformation/calibration
        - score_out: function that transforms scores
    """

    assert classifier_type in ['monoton', 'unrestricted']
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if classifier_type == 'monoton':

        # Define Model that predicts probability of failure from given score
        class Model(nn.Module):
            """
            Model that predicts probability of failure from given score
            The model is by construction monotonous
            """
            def __init__(self):
                super().__init__()

                self.model1 = monotonous_net.Net(n_blocks=2, dim=1, ls=128, n_condim=0)

            def forward(self, x):
                out =  torch.cat( (self.model1(x[:,:1], y=None), -self.model1(x[:,:1], y=None)),1)
                return out

        model = Model().to(device)
        optimizer= torch.optim.Adam(model.parameters(), lr=LR)
        loss_fct = nn.CrossEntropyLoss()
        batch_size = BATCH_SIZE

        scores_train = torch.from_numpy(scores_train)
        true_false_train = torch.from_numpy(true_false_train)

        # Train Model
        for _ in range(5_000):

            optimizer.zero_grad()

            idx = torch.randperm(true_false_train.shape[0])[:batch_size]
            out = model(scores_train[idx].view(-1,1).to(device))

            loss = loss_fct(out, true_false_train[idx].to(device).long())
            loss.backward()

            optimizer.step()

        # Define new Score function (given through transformed scores)
        def score_out(x):
            with torch.no_grad():
                return F.softmax(model(torch.from_numpy(x).to(device).view(-1,1)), dim=1)[:,0].view(-1).cpu().numpy() 

        return score_out(scores_train.numpy()), score_out

    if classifier_type == 'unrestricted':

        # Unrestricted model that predicts probability of failure from given score
        model = nn.Sequential(nn.Linear(1,1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),nn.Linear(1024,2)).to(device)

        optimizer= torch.optim.Adam(model.parameters(), lr=LR)
        loss_fct = nn.CrossEntropyLoss()
        batch_size = BATCH_SIZE

        scores_train = torch.from_numpy(scores_train)
        true_false_train = torch.from_numpy(true_false_train)

        # Train Model
        for _ in range(5_000):

            optimizer.zero_grad()

            idx = torch.randperm(true_false_train.shape[0])[:batch_size]
            out = model(scores_train[idx].view(-1,1).to(device).float())

            loss = loss_fct(out, true_false_train[idx].to(device).long())
            loss.backward()

            optimizer.step()

        # Define new Score function (given through transformed scores)
        def score_out(x):
            with torch.no_grad():
                return F.softmax(model(torch.from_numpy(x).to(device).view(-1,1).float()), dim=1)[:,0].view(-1).cpu().numpy() 

        return score_out(scores_train.numpy()), score_out


