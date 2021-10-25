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
        #print("softmax loss first part")
        affines = features.matmul(self.weights.t()) + self.bias
        logits = affines
        # The temperature may be calibrated after training for improved predictive uncertainty estimation
        return logits / self.temperature


class SoftMaxLossSecondPart(nn.Module):
    def __init__(self, model_classifier):
        super(SoftMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, debug=False):
        #print("softmax loss second part")
        loss = self.loss(logits, targets)
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(self.model_classifier.weights.size(0))[targets].long().cuda()
            intra_inter_logits = torch.where(targets_one_hot != 0, logits[:len(targets)], torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits[:len(targets)])
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]
            distance_scale = 1
            return loss, distance_scale, intra_logits, inter_logits

