#!/bin/bash
maxd=13
i=0
cuda=4

for (( hidden_dims=3; hidden_dims< $maxd; hidden_dims+=$cuda))
  do
    {
      for (( i=0; i< $cuda; i+=1))
      do
        {
          echo "process n_hidden $(($hidden_dims+$i))"
          CUDA_VISIBLE_DEVICES=$i python 01_get_representations.py --hidden_dims $(($hidden_dims+$i)) --input_path "./dataset/" --filename_train "datatrain.pkl" --filename_valid "datavalid.pkl" --filename_test "datatest.pkl" --filename_data "data.pkl" --input_dim 26 --n_layers 2 --epoch 300 >"./runs/$(($hidden_dims+$i)).log"
        }&
      done
      wait
   }&
  done
