#!/bin/bash

train=$1
data_path=$2

# Hyperparameters to be changed - please do not edit the placeholders
hidden_sizes="HIDDEN"
middle_dropout=MIDDLE_DROPOUT
last_dropout=LAST_DROPOUT
weight_decay=1e-06

{
tstart=`date +%s.%N`
date1=`date`
echo $date1

python $train \
  --x cls_T11_x_processed.npz \
  --y cls_T10_y_processed.npz \
  --folding cls_T11_fold_vector_processed.npy \
  --weights_class cls_weights.csv \
  --hidden_sizes $hidden_sizes \
  --weight_decay $weight_decay \
  --dropouts_trunk $middle_dropout $last_dropout\
  --last_non_linearity relu \
  --non_linearity relu \
  --input_transform none \
  --lr 0.001 \
  --lr_alpha 0.3 \
  --lr_steps 10 \
  --epochs 100 \
  --normalize_loss 100_000 \
  --eval_frequency 1 \
  --batch_ratio 0.02 \
  --fold_va 4 \
  --fold_te 0 \
  --verbose 1 \
  --profile 1 \
  --save_model 0 \
  --run_name local_cls_HP_scan

tend=`date +%s.%N`
date2=`date`

echo $date2

awk -v tstart=$tstart -v tend=$tend 'BEGIN{time=tend-tstart; printf "TIME [s]: %12.2f\n", time}'
} > out1.dat 2> out2.dat

