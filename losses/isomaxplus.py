import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxPlusLossFirstPart(nn.Module):
    """Replaces the model classifier last layer nn.Linear()"""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        self.temperature = temperature
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances = torch.abs(self.distance_scale) * F.pairwise_distance(
            F.normalize(features).unsqueeze(2), F.normalize(self.prototypes).t().unsqueeze(0), p=2.0)
        logits = -distances
        # The temperature may be calibrated after training for improved predictive uncertainty estimation
        return logits / self.temperature


class IsoMaxPlusLossSecondPart(nn.Module):
    """Replaces the nn.CrossEntropyLoss()"""
    def __init__(self, model_classifier):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = 10.0

    def forward(self, logits, targets, debug=False):
        #############################################################################
        #############################################################################
        """Probabilities and logarithms are calculate separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        #############################################################################
        #############################################################################
        distances = -logits
        distance_scale = torch.abs(self.model_classifier.distance_scale)
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(self.model_classifier.prototypes.size(0))[targets].long().cuda()
            intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
            inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
            intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
            inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
            return loss, distance_scale.item(), intra_distances, inter_distances
