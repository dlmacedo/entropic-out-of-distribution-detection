import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxIsometricLossFirstPart(nn.Module):
    """Replaces the model classifier last layer nn.Linear()"""
    def __init__(self, num_features, num_classes):
        super(IsoMaxIsometricLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances = F.pairwise_distance(F.normalize(features).unsqueeze(2), F.normalize(self.prototypes).t().unsqueeze(0), p=2.0)
        logits = -torch.abs(self.distance_scale) * distances
        #print("isometric isomax loss first part")
        return logits


class IsoMaxIsometricLossSecondPart(nn.Module):
    """Replaces the nn.CrossEntropyLoss()"""
    def __init__(self, model_classifier):
        super(IsoMaxIsometricLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = 10.0

    def forward(self, logits, targets, debug=False):
        ############################################################################
        """Logarithms and probabilities are calculate separately and sequentially"""
        """Hence, nn.CrossEntropyLoss() must not be used!!!"""
        ############################################################################
        probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * logits)
        probabilities_at_targets = probabilities_for_training[range(logits.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        #print("isometric isomax loss second part")
        if not debug:
            return loss
        else:
            print("distance scale: {0:.8f}".format(torch.abs(self.model_classifier.distance_scale).item()))
            targets_one_hot = torch.eye(self.model_classifier.prototypes.size(0))[targets].long().cuda()
            intra_inter_logits = torch.where(targets_one_hot != 0, -logits, torch.Tensor([float('Inf')]).cuda())
            inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), -logits)
            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]
            #cls_probabilities = nn.Softmax(dim=1)(self.entropic_scale * logits)
            #ood_probabilities = nn.Softmax(dim=1)(logits)
            #max_logits = logits.max(dim=1)[0]
            #return loss, cls_probabilities, ood_probabilities, max_logits, intra_logits, inter_logits
            return loss, intra_logits, inter_logits
