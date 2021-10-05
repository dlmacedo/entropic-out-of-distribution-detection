# Code reused from: https://github.com/kuangliu/pytorch-cifar
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import net
import losses
import tools
from torchmetrics import AUROC
import random
import numpy
import torchnet as tnt


base_seed = 42
random.seed(base_seed)
numpy.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed(base_seed)
cudnn.benchmark = False
cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch one

# Data
print('==> Preparing data...')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),])

trainset = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=4,
    worker_init_fn=lambda worker_id: random.seed(base_seed + worker_id),
    )
testset = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=4,
    )

# Model
print('==> Building model...')
model = net.DenseNet3(100, 10)
model = model.to(device)

##################################################################
#criterion = nn.CrossEntropyLoss()
criterion = losses.IsoMaxPlusLossSecondPart(model.classifier)
##################################################################

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1*1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200, 250], gamma=0.1)


def train(epoch):
    print('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        tools.progress_bar(batch_idx, len(trainloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tools.progress_bar(batch_idx, len(testloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving...')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pth')
        best_acc = acc


def detect(inloader, oodloader):
    auroc = AUROC(pos_label=1)
    auroctnt = tnt.meter.AUCMeter()
    model.eval()
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(inloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets.fill_(1)
            outputs = model(inputs)
            #probabilities = torch.nn.Softmax(dim=1)(outputs)
            #score = probabilities.max(dim=1)[0] # this is the maximum probability score 
            #entropies = -(probabilities * torch.log(probabilities)).sum(dim=1)
            #score = -entropies # this is the negative entropy score
            # the negative entropy score is the best option for the IsoMax loss
            # outputs are equal to logits, which in turn are equivalent to negative distances
            score = outputs.max(dim=1)[0] # this is the minimum distance score
            # the minimum distance score is the best option for the IsoMax+ loss
            auroc.update(score, targets) 
            auroctnt.add(score, targets)           
        for _, (inputs, targets) in enumerate(oodloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets.fill_(0)
            outputs = model(inputs)
            #probabilities = torch.nn.Softmax(dim=1)(outputs)
            #score = probabilities.max(dim=1)[0] # this is the maximum probability score 
            #entropies = -(probabilities * torch.log(probabilities)).sum(dim=1)
            #score = -entropies # this is the negative entropy score
            # the negative entropy score is the best option for the IsoMax loss
            # outputs are equal to logits, which in turn are equivalent to negative distances
            score = outputs.max(dim=1)[0] # this is the minimum distance score for detection
            # the minimum distance score is the best option for the IsoMax+ loss
            auroc.update(score, targets)            
            auroctnt.add(score, targets)            
    return auroc.compute(), auroctnt.value()[0]

total_epochs = 300
for epoch in range(start_epoch, start_epoch + total_epochs):
    print()
    for param_group in optimizer.param_groups:
        print("LEARNING RATE: ", param_group["lr"])
    train(epoch)
    test(epoch)
    scheduler.step()

checkpoint = torch.load('checkpoint/ckpt.pth')
model.load_state_dict(checkpoint['model'])
test_acc = checkpoint['acc']

print()
print("###################################################")
print("Test Accuracy (%): {0:.4f}".format(test_acc))
print("###################################################")
print()

dataroot = os.path.expanduser(os.path.join('data', 'Imagenet_resize'))
oodset = torchvision.datasets.ImageFolder(dataroot, transform=transform_test)
oodloader = torch.utils.data.DataLoader(oodset, batch_size=64, shuffle=False, num_workers=4)
auroc = detect(testloader, oodloader)
print()
print("#################################################################################################################")
print("Detection performance for ImageNet Resize as Out-of-Distribution [AUROC] (%): {0:.4f}".format(100. * auroc[0].item()), auroc[1])
print("#################################################################################################################")
print()

dataroot = os.path.expanduser(os.path.join('data', 'LSUN_resize'))
oodset = torchvision.datasets.ImageFolder(dataroot, transform=transform_test)
oodloader = torch.utils.data.DataLoader(oodset, batch_size=64, shuffle=False, num_workers=4)
auroc = detect(testloader, oodloader)
print()
print("#################################################################################################################")
print("Detection performance for LSUN Resize as Out-of-Distribution [AUROC] (%): {0:.4f}".format(100. * auroc[0].item()), auroc[1])
print("#################################################################################################################")
print()

oodset = torchvision.datasets.SVHN(root='data/svhn', split="test", download=True, transform=transform_test)
oodloader = torch.utils.data.DataLoader(oodset, batch_size=64, shuffle=False, num_workers=4)
auroc = detect(testloader, oodloader)
print()
print("#################################################################################################################")
print("Detection performance for SVHN as Out-of-Distribution [AUROC] (%): {0:.4f}".format(100. * auroc[0].item()), auroc[1])
print("#################################################################################################################")
print()
