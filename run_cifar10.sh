#!/usr/bin/env bash

python main.py -ei="ood" -et "train_classify" -ec "data~cifar10+model~densenetbc100+loss~softmax_no_no_no_final" -gpu 0 -x 5
python main.py -ei="ood" -et "train_classify" -ec "data~cifar10+model~densenetbc100+loss~isomax_no_no_no_final" -gpu 0 -x 5
python main.py -ei="ood" -et "train_classify" -ec "data~cifar10+model~densenetbc100+loss~isomaxplus_no_no_no_final" -gpu 0 -x 5
python main.py -ei="ood" -et "train_classify" -ec "data~cifar10+model~resnet110+loss~softmax_no_no_no_final" -gpu 0 -x 5
python main.py -ei="ood" -et "train_classify" -ec "data~cifar10+model~resnet110+loss~isomax_no_no_no_final" -gpu 0 -x 5
python main.py -ei="ood" -et "train_classify" -ec "data~cifar10+model~resnet110+loss~isomaxplus_no_no_no_final" -gpu 0 -x 5

python detect.py --dir ood --dataset cifar10 --net_type densenetbc100 --loss softmax_no_no_no_final --gpu 0 -x 5
python detect.py --dir ood --dataset cifar10 --net_type densenetbc100 --loss isomax_no_no_no_final --gpu 0 -x 5
python detect.py --dir ood --dataset cifar10 --net_type densenetbc100 --loss isomaxplus_no_no_no_final --gpu 0 -x 5
python detect.py --dir ood --dataset cifar10 --net_type resnet110 --loss softmax_no_no_no_final --gpu 0 -x 5
python detect.py --dir ood --dataset cifar10 --net_type resnet110 --loss isomax_no_no_no_final --gpu 0 -x 5
python detect.py --dir ood --dataset cifar10 --net_type resnet110 --loss isomaxplus_no_no_no_final --gpu 0 -x 5
