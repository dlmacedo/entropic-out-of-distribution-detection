import os
import pickle
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch
import torch.nn.functional as F
import csv
import numpy as np
from sklearn import metrics


def compute_weights(iterable):
    return [sum(iterable) / (iterable[i] * len(iterable)) if iterable[i] != 0 else float("inf") for i in range(len(iterable))]


def print_format(iterable):
    return ["{0:.8f}".format(i) if i is not float("inf") else "{0}".format(i) for i in iterable]


def probabilities(outputs):
    return F.softmax(outputs, dim=1)


def max_probabilities(outputs):
    return F.softmax(outputs, dim=1).max(dim=1)[0]


def predictions(outputs):
    return outputs.argmax(dim=1)


def predictions_total(outputs):
    return outputs.argmax(dim=1).bincount(minlength=outputs.size(1)).tolist()


def entropies(outputs):
    probabilities_log_probabilities = F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)
    return -1.0 * probabilities_log_probabilities.sum(dim=1)


def entropies_grads(outputs):
    entropy_grads = - (1.0 + F.log_softmax(outputs, dim=1))
    return entropy_grads.sum(dim=0).tolist()


def cross_entropies(outputs, targets):
    return - 1.0 * F.log_softmax(outputs, dim=1)[range(outputs.size(0)), targets]


def cross_entropies_grads(outputs, targets):
    cross_entropies_grads = [0 for i in range(outputs.size(1))]
    for i in range(len(predictions(outputs))):
        cross_entropies_grads[predictions(outputs)[i]] += - (1.0 - (F.softmax(outputs, dim=1)[i, targets[i]].item()))
    return cross_entropies_grads


def make_equitable(outputs, criterion, weights):
    weights = torch.Tensor(weights).cuda()
    weights.requires_grad = False
    return weights[predictions(outputs)] * criterion[range(outputs.size(0))]


def entropies_from_logits(logits):
    return -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)


def entropies_from_probabilities(probabilities):
    if len(probabilities.size()) == 2:
        return -(probabilities * torch.log(probabilities)).sum(dim=1)
    elif len(probabilities.size()) == 3:
        return -(probabilities * torch.log(probabilities)).sum(dim=2).mean(dim=1)


def save_object(object, path, file):
    with open(os.path.join(path, file + '.pkl'), 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_object(path, file):
    with open(os.path.join(path, file + '.pkl'), 'rb') as f:
        return pickle.load(f)


def save_dict_list_to_csv(dict_list, path, file):
    with open(os.path.join(path, file + '.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict_list[0].keys())
        writer.writeheader()
        for dict in dict_list:
            writer.writerow(dict)


def load_dict_list_from_csv(path, file):
    dict_list = []
    with open(os.path.join(path, file + '.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for dict in reader:
            dict_list.append(dict)
    return dict_list


class MeanMeter(object):
    """computes and stores the current averaged current mean"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def purity(y_true, y_pred):
    """compute contingency matrix (also called confusion matrix)"""
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def asinh(x):
    return torch.log(x+(x**2+1)**0.5)


def acosh(x):
    return torch.log(x+(x**2-1)**0.5)


def atanh(x):
    return 0.5*torch.log(((1+x)/((1-x)+0.000001))+0.000001)


def sinh(x):
    return (torch.exp(x)-torch.exp(-x))/2


def cosine_similarity(features, prototypes):
    return F.cosine_similarity(features.unsqueeze(2), prototypes.t().unsqueeze(0), dim=1, eps=1e-6)


def euclidean_distances(features, prototypes, pnorm):
    if len(prototypes.size()) == 2:
        return F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=pnorm)
    else:
        return F.pairwise_distance(features.unsqueeze(2).unsqueeze(3), prototypes.unsqueeze(0).permute(0, 3, 1, 2), p=pnorm)


def mahalanobis_distances(features, prototypes, precisions):
    diff = features.unsqueeze(2) - prototypes.t().unsqueeze(0)
    diff2 = features.t().unsqueeze(0) - prototypes.unsqueeze(2)
    precision_diff = torch.matmul(precisions.unsqueeze(0), diff)
    extended_product = torch.matmul(diff2.permute(2, 0, 1), precision_diff)
    mahalanobis_square = torch.diagonal(extended_product, offset=0, dim1=1, dim2=2)
    mahalanobis = torch.sqrt(mahalanobis_square)
    return mahalanobis


def multiprecisions_mahalanobis_distances(features, prototypes, multiprecisions):
    mahalanobis_square = torch.Tensor(features.size(0), prototypes.size(0)).cuda()
    for prototype in range(prototypes.size(0)):
        diff = features - prototypes[prototype]
        multiprecisions.unsqueeze(0)
        diff.unsqueeze(2)
        precision_diff = torch.matmul(multiprecisions.unsqueeze(0), diff.unsqueeze(2))
        product = torch.matmul(diff.unsqueeze(1), precision_diff).squeeze()
        mahalanobis_square[:, prototype] = product
    mahalanobis = torch.sqrt(mahalanobis_square)
    return mahalanobis


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    #print("calling randbox")
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    """
    r = 0.5 + np.random.rand(1)/2
    s = 0.5/r
    if np.random.rand(1) < 0.5:
        r, s = s, r
    cut_w = np.int(W * r)
    cut_h = np.int(H * s)
    """
    #cx = np.random.randint(W)
    #cy = np.random.randint(H)
    cx = np.random.randint(cut_w // 2, high=W - cut_w // 2)
    cy = np.random.randint(cut_h // 2, high=H - cut_h // 2)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("total number of parameters: {:.2e}".format(total_num_params))
