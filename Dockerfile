FROM pytorch/pytorch:0.4_cuda9_cudnn7
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN pip install numpy scipy matplotlib librosa==0.6.0 tensorflow tensorboardX inflect==0.2.5 Unidecode==1.0.22 jupyter tqdm
