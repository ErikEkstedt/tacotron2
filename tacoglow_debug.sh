#!/bin/bash

sudo nvidia-docker run -it \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
  -v /projects/tacotron/tacotron2/:/workspace \
  -v /home/erikekst/Data/tacotron:/home/erikekst/Data/tacotron \
  tacotron python -m pdb train.py \
  --output_directory=debug_outdir \
  --log_directory=logdir \
  --hparams=max_decoder_steps=100 \
  #--hparams=distributed_run=True
