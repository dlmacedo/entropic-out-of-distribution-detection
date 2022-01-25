#!/bin/bash

echo "Begining..."

# download and unzip dataset
wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
tar xvf Imagenet_resize.tar.gz
tar xvf LSUN_resize.tar.gz
rm Imagenet_resize.tar.gz
rm LSUN_resize.tar.gz

echo "Done!!!"
