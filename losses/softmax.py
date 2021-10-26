import torch.nn as nn
import torch
import math


class SoftMaxLossFirstPart(nn.Module):
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(SoftMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.temperature = temperature
        nn.init.uniform_(self.weights, a=-math.sqrt(1/self.num_features), b=math.sqrt(1/self.num_features))
        nn.init.zeros_(self.bias)

    def forward(self, features):
        logits = features.matmul(self.weights.t()) + self.bias
        # The temperature may be calibrated after training for improved predictive uncertainty estimation
        return logits / self.temperature


class SoftMaxLossSecondPart(nn.Module):
    def __init__(self, model_classifier):
        super(SoftMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, debug=False):
        loss = self.loss(logits, targets)
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(self.model_classifier.weights.size(0))[targets].long().cuda()
            intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]
            return loss, 1.0, intra_logits, inter_logits
