# parameters taken from the Hyperparameter Optimization process (trial #1)
python train.py \
    --epochs 200 \
    --resnet2d-inplanes 32 40 39 \
    --resnet3d-inplanes 18 27 20 \
    --batch-size 4 \
    --optimizer Adam \
    --learning-rate 0.00032117685902790307 \
    --weight-decay 0.0012959406987438113 \
    --device cuda:1