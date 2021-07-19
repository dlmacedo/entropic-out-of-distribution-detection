from __future__ import print_function
import argparse
import torch
import models
import os
import losses
import data_loader
import calculate_log as callog
from torchvision import transforms

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--loss', required=True, help='the loss used')
parser.add_argument('--dir', default="", type=str, help='Part of the dir to use')
parser.add_argument('-x', '--executions', default=1, type=int, metavar='N', help='Number of executions (default: 1)')

args = parser.parse_args()
print(args)
torch.cuda.set_device(args.gpu)


def main():
    dir_path = os.path.join("experiments", args.dir, "train_classify", "data~"+args.dataset+"+model~"+args.net_type+"+loss~"+str(args.loss))
    file_path = os.path.join(dir_path, "results_odd.csv")

    with open(file_path, "w") as results_file:
        results_file.write(
            "EXECUTION,MODEL,IN-DATA,OUT-DATA,LOSS,AD-HOC,SCORE,INFER-LEARN,INFER-TRANS,"
            "TNR,AUROC,DTACC,AUIN,AUOUT,CPU_FALSE,CPU_TRUE,GPU_FALSE,GPU_TRUE,TEMPERATURE,MAGNITUDE\n")

    args_outf = os.path.join("temporary", args.dir, args.loss, args.net_type + '+' + args.dataset)
    if os.path.isdir(args_outf) == False:
        os.makedirs(args_outf)
    
    # define number of classes
    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'imagenet32':
        args.num_classes = 1000
    else:
        args.num_classes = 10

    if args.dataset == 'cifar10':
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
    elif args.dataset == 'cifar100':
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
    elif args.dataset == 'svhn':
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']

    if args.dataset == 'cifar10':
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])
    elif args.dataset == 'cifar100':
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))])
    elif args.dataset == 'svhn':
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.437, 0.443, 0.472), (0.198, 0.201, 0.197))])

    for args.execution in range(1, args.executions + 1):    
        print("EXECUTION:", args.execution)
        pre_trained_net = os.path.join(dir_path, "model" + str(args.execution) + ".pth")

        if args.loss.split("_")[0] == "softmax":
            loss_first_part = losses.SoftMaxLossFirstPart
            scores = ["ES"]
        elif args.loss.split("_")[0] == "isomax":
            loss_first_part = losses.IsoMaxLossFirstPart
            scores = ["ES"]
        elif args.loss.split("_")[0] == "isomaxisometric":
            loss_first_part = losses.IsoMaxIsometricLossFirstPart
            scores = ["MDS"]

        # load networks
        if args.net_type == 'densenetbc100':
            model = models.DenseNet3(100, int(args.num_classes), loss_first_part=loss_first_part)
        elif args.net_type == 'resnet110':
            model = models.ResNet110(num_c=args.num_classes, loss_first_part=loss_first_part)
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        model.cuda()
        print('load model: ' + args.net_type)
        
        # load dataset
        print('load target valid data: ', args.dataset)
        _, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)

        for score in scores:
            print("\n\n\n###############################")
            print("###############################")
            print("SCORE:", score)
            print("###############################")
            print("###############################")
            base_line_list = []
            get_scores(model, test_loader, args_outf, True, score)
            out_count = 0
            for out_dist in out_dist_list:
                out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot)
                print('Out-distribution: ' + out_dist)
                get_scores(model, out_test_loader, args_outf, False, score)
                test_results = callog.metric(args_outf, ['PoT'])
                base_line_list.append(test_results)
                out_count += 1
            
            # print the results
            mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
            print('Baseline method: train in_distribution: ' + args.dataset + '==========')
            count_out = 0
            for results in base_line_list:
                print('out_distribution: '+ out_dist_list[count_out])
                for mtype in mtypes:
                    print(' {mtype:6s}'.format(mtype=mtype), end='')
                print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
                print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
                print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
                print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
                print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
                print('')
                #Saving odd results:
                with open(file_path, "a") as results_file:
                    results_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        str(args.execution), args.net_type, args.dataset, out_dist_list[count_out],
                        str(args.loss), "NATIVE", score, 'NO', False,
                        '{:.2f}'.format(100.*results['PoT']['TNR']),
                        '{:.2f}'.format(100.*results['PoT']['AUROC']),
                        '{:.2f}'.format(100.*results['PoT']['DTACC']),
                        '{:.2f}'.format(100.*results['PoT']['AUIN']),
                        '{:.2f}'.format(100.*results['PoT']['AUOUT']),
                        0, 0, 0, 0, 1, 0))
                count_out += 1


def get_scores(model, test_loader, outf, out_flag, score_type=None):
    model.eval()
    total = 0
    if out_flag == True:
        temp_file_name_val = '%s/confidence_PoV_In.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt'%(outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt'%(outf)     
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')

    for data, _ in test_loader:
        total += data.size(0)
        data = data.cuda()  
        with torch.no_grad():
            logits = model(data)
            probabilities = torch.nn.Softmax(dim=1)(logits)
            if score_type == "MPS": # the maximum probability score
                soft_out = probabilities.max(dim=1)[0]
            elif score_type == "ES": # the negative entropy score
                soft_out = (probabilities * torch.log(probabilities)).sum(dim=1)
            elif score_type == "MDS": # the minimum distance score
                soft_out = logits.max(dim=1)[0]
        for i in range(data.size(0)):
            f.write("{}\n".format(soft_out[i]))
    f.close()
    g.close()


if __name__ == '__main__':
    main()
