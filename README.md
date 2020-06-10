# DACT

We've created DACT, a new algorithm for achieving adaptive computation time that, unlike existing approaches, is fully differentiable and can work in conjunction with complex models.
DACT replaces hard limits and piecewise functions with inductive biases to allow the network to choose during evaluation the amount of computation needed for the current input.
The resulting models learn the tradeoff between precision and complexity and actively adapt their architectures accordingly.
Our paper shows that, when applied to the widely known MAC architecture and the visual reasoning task, DACT can improve interpretability, make the model more robust to relevant hyperparameter changes, all while increasing the performance to computation ratio.

[LINK TO BLOG](https://ceyzaguirre4.github.io/DACT-for-Visual-Reasoning/)

[LINK TO PREPRINT](http://arxiv.org/abs/2004.12770)


# Code for DACT-MAC

Forked from [ceyzaguirre4/mac-network-pytorch](https://github.com/ceyzaguirre4/mac-network-pytorch) which is based on  [Memory, Attention and Composition (MAC) Network for CLEVR from Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067).

## 1. Install requirements

Run the following command to install missing dependencies:

~~~bash
pip install -r requirements.txt
~~~

The dependencies include `comet_ml` for metric reporting, but all experiments can be run without installing it by carefully commenting the appropiate lines.


## 2. Preprocess questions and images

### 2.1 CLEVR

~~~bash
python3 preprocess.py [CLEVR directory]
python3 image_feature.py [CLEVR directory]
~~~

### 2.1 GQA

For GQA we use the object-based features for GQA, extracted from faster-RCNN available in the [oficial website](https://cs.stanford.edu/people/dorarad/gqa/download.html) and extract the hdf5 files to `[GQA directory]/features/objects`.

~~~bash
# `all` to use all questions (as in paper), `balanced`to only use oficial balanced subset.
python3 preprocess_GQA.py [GQA directory] [all | balanced]
~~~

## 3. Train the model

### 3.1 CLEVR

Training the model with all-default hyper-parameters trains a 12-step MAC without gating or self-attention for 10 epochs on *CLEVR*.

~~~bash
python3 train.py DATALOADER.FEATURES_PATH [CLEVR directory]
~~~

Alternatively, pass the desired configuration file as a parameter.

~~~bash
python3 train.py --config-file=[path to config file] DATALOADER.FEATURES_PATH [CLEVR directory]
~~~

Configuration files to replicate results from paper are provided in the `configs/` directory. 
All (CLEVR) adaptive models require a trained 12 step MAC from which to load pre-trained weights.
For instance, to train DACT with ponder cost `5e-3` do:

~~~bash
# pretrain 12-step MAC
python3 train.py --config-file=configs/CLEVR/MAC/mac12.yaml DATALOADER.FEATURES_PATH [CLEVR directory]
# OR use default params
# python3 train.py DATALOADER.FEATURES_PATH [CLEVR directory]

# train DACT
python3 train.py --config-file=configs/CLEVR/DACT/ours_0005.yaml DATALOADER.FEATURES_PATH [CLEVR directory]
~~~

The same is valid for all gated MACs; to train gated the gated variants provided in the configuration files do:

~~~bash
# pretrain 3-step MAC
python3 train.py --config-file=configs/CLEVR/MAC/mac3.yaml DATALOADER.FEATURES_PATH [CLEVR directory]

# train gated variant from pretrained weights
python3 train.py --config-file=configs/CLEVR/MAC/mac3+gate.yaml DATALOADER.FEATURES_PATH [CLEVR directory]
~~~

### 3.2 GQA

Training on *GQA dataset* is achieved by using the `--mode` argument:


For instance, training the model with all-default hyper-parameters in `--mode=gqa` trains a 4-step MAC without gating or self-attention for 5 epochs:

~~~bash
python3 train.py --mode=gqa DATALOADER.FEATURES_PATH [GQA directory]
~~~

All (GQA) adaptive models require a trained 4 step MAC from which to load pre-trained weights.
For instance, to train DACT with ponder cost `5e-3` do:

~~~bash
# pretrain 4-step MAC
python3 train.py --mode=gqa --config-file=configs/GQA/MAC/mac12.yaml DATALOADER.FEATURES_PATH [GQA directory]
# OR use default params
# python3 train.py --mode=gqa DATALOADER.FEATURES_PATH [GQA directory]

# train DACT
python3 train.py --mode=gqa --config-file=configs/GQA/DACT/ours_0005.yaml DATALOADER.FEATURES_PATH [GQA directory]
~~~


## Cite

~~~
@InProceedings{Eyzaguirre_2020_CVPR,
  author = {Eyzaguirre, Cristobal and Soto, Alvaro},
  title = {Differentiable Adaptive Computation Time for Visual Reasoning},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
~~~