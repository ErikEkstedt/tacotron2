#!/bin/bash

sudo nvidia-docker run -it \
  --ipc=host \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
  -v /projects/tacotron/tacotron2/:/workspace \
  -v /home/erikekst/Data/tacotron:/home/erikekst/Data/tacotron \
  tacotron python -m multiproc train.py \
  --output_directory=outdir \
  --log_directory=logdir \
  --hparams=iters_per_checkpoint=100,distributed_run=True
