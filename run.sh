#!/bin/bash

# Market-1501
python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-softmax-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss softmax --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-triplet-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss triplet --weight-x 0 --batch-size 64 --train-sampler RandomIdentitySampler --stepsize 5 20 40 --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-softmax+triplet-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss triplet --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-center-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss center --weight-t 0.00003 --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-npair-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss n-pair --train-sampler RandomIdentitySampler --num-instances 2 --batch-size 64 --dist-metric cosine --eval-freq 10

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-histogram-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss histogram --train-sampler RandomIdentitySampler --num-instances 2 --batch-size 64 --stepsize 2 20 40 --dist-metric cosine --eval-freq 10

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-focal-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss focal --dist-metric cosine --eval-freq 10

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-ms-cosine  --gpu-devices 4,5,6,7 -a resnet50 --loss ms --train-sampler RandomIdentitySampler --batch-size 160 --num-instances 5 --eval-freq 10 --lr 0.0001 --stepsize 20 40 --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-sphere_no_margin-cosine --gpu-devices 4,5,6,7 -a resnet50_angularlinear --loss a-softmax --dist-metric cosine --eval-freq 10

#After modified sphere.py   nomargin-->margin
#python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-sphere_no_margin-cosine --gpu-devices 4,5,6,7 -a resnet50_angularlinear --loss a-softmax --dist-metric cosine --eval-freq 10

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-cosface-cosine  --gpu-devices 4,5,6,7 -a resnet50_cosface --loss cosface --eval-freq 10 --dist-metric cosine --stepsize 5 20 40

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s market1501 --save-dir log/market/resnet50-market-arcface-cosine  --gpu-devices 4,5,6,7 -a resnet50_arcface --loss arcface --eval-freq 10 --dist-metric cosine --stepsize 5 20 40



# DukeMTMC-reid
python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-softmax-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss softmax --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-triplet-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss triplet --weight-x 0 --batch-size 64 --train-sampler RandomIdentitySampler --stepsize 5 20 40 --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-softmax+triplet-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss triplet --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-center-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss center --weight-t 0.00001 --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-npair-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss n-pair --train-sampler RandomIdentitySampler --num-instances 2 --batch-size 64 --dist-metric cosine --eval-freq 10

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-histogram-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss histogram --train-sampler RandomIdentitySampler --num-instances 2 --batch-size 64 --stepsize 2 20 40 --dist-metric cosine --eval-freq 10

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-focal-cosine --gpu-devices 4,5,6,7 -a resnet50 --loss focal --dist-metric cosine --eval-freq 10

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-ms-cosine  --gpu-devices 4,5,6,7 -a resnet50 --loss ms --train-sampler RandomIdentitySampler --batch-size 160 --num-instances 5 --eval-freq 10 --lr 0.0001 --stepsize 20 40 --dist-metric cosine

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-sphere_no_margin-cosine --gpu-devices 4,5,6,7 -a resnet50_angularlinear --loss a-softmax --dist-metric cosine --eval-freq 10

#After modified sphere.py   nomargin-->margin
#python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-sphere_no_margin-cosine --gpu-devices 4,5,6,7 -a resnet50_angularlinear --loss a-softmax --dist-metric cosine --eval-freq 10

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/market/resnet50-duke-cosface-cosine  --gpu-devices 4,5,6,7 -a resnet50_cosface --loss cosface --eval-freq 10 --dist-metric cosine --stepsize 5 20 40

python scripts/main.py --root /home/wangyuxin/data/Code/deep-person-reid/reid-data/ -s dukemtmcreid --save-dir log/duke/resnet50-duke-arcface-cosine  --gpu-devices 4,5,6,7 -a resnet50_arcface --loss arcface --eval-freq 10 --dist-metric cosine --stepsize 5 20 40











