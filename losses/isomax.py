import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxLossFirstPart(nn.Module):
    """Replaces the model classifier last layer nn.Linear()"""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.temperature = temperature
        nn.init.constant_(self.prototypes, 0.0)

    def forward(self, features):
        #print("isomax loss first part")
        distances = F.pairwise_distance(features.unsqueeze(2), self.prototypes.t().unsqueeze(0), p=2.0)
        logits = -distances
        # The temperature may be calibrated after training for improved predictive uncertainty estimation
        return logits / self.temperature


class IsoMaxLossSecondPart(nn.Module):
    """Replaces the nn.CrossEntropyLoss()"""
    def __init__(self, model_classifier):
        super(IsoMaxLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = 10.0

    def forward(self, logits, targets, debug=False):
        #############################################################################
        #############################################################################
        """Probabilities and logarithms are calculate separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
        #############################################################################
        #print("isomax loss second part")
        probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * logits[:len(targets)])
        probabilities_at_targets = probabilities_for_training[range(logits.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(self.model_classifier.prototypes.size(0))[targets].long().cuda()
            intra_inter_logits = torch.where(targets_one_hot != 0, -logits[:len(targets)], torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), -logits[:len(targets)])
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]
            distance_scale = 1
            return loss, distance_scale, intra_logits, inter_logits

