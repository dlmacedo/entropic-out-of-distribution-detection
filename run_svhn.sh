#!/usr/bin/env bash

python main.py -ei="ood" -et "train_classify" -ec "data~svhn+model~densenetbc100+loss~softmax_no_no_no_final" -gpu 0 -x 10
python main.py -ei="ood" -et "train_classify" -ec "data~svhn+model~densenetbc100+loss~isomax_no_no_no_final" -gpu 0 -x 10
python main.py -ei="ood" -et "train_classify" -ec "data~svhn+model~densenetbc100+loss~isomaxplus_no_no_no_final" -gpu 0 -x 10
python main.py -ei="ood" -et "train_classify" -ec "data~svhn+model~resnet110+loss~softmax_no_no_no_final" -gpu 0 -x 10
python main.py -ei="ood" -et "train_classify" -ec "data~svhn+model~resnet110+loss~isomax_no_no_no_final" -gpu 0 -x 10
python main.py -ei="ood" -et "train_classify" -ec "data~svhn+model~resnet110+loss~isomaxplus_no_no_no_final" -gpu 0 -x 10

python detect.py --dir ood --dataset svhn --net_type densenetbc100 --loss softmax_no_no_no_final --gpu 0 -x 10
python detect.py --dir ood --dataset svhn --net_type densenetbc100 --loss isomax_no_no_no_final --gpu 0 -x 10
python detect.py --dir ood --dataset svhn --net_type densenetbc100 --loss isomaxplus_no_no_no_final --gpu 0 -x 10
python detect.py --dir ood --dataset svhn --net_type resnet110 --loss softmax_no_no_no_final --gpu 0 -x 10
python detect.py --dir ood --dataset svhn --net_type resnet110 --loss isomax_no_no_no_final --gpu 0 -x 10
python detect.py --dir ood --dataset svhn --net_type resnet110 --loss isomaxplus_no_no_no_final --gpu 0 -x 10
