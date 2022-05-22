import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.constant_(self.prototypes, 0.0)

    def forward(self, features):
        distances = torch.cdist(features, self.prototypes, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature


class IsoMaxLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):
        #############################################################################
        #############################################################################
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
        #############################################################################
        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(distances.size(1))[targets].long().cuda()
            intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
            inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
            intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
            inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
            return loss, 1.0, intra_distances, inter_distances

