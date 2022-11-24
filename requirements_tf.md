## Requirements
```shell
# Create a virtual environment with anaconda3
conda create -n tf2 python=3.8
conda activate tf2

# tf2 nightly and iree
python -m pip install tf-nightly
python -m pip install \
		iree-compiler \
		iree-runtime \
		iree-tools-tf

# For DLRM
pip install functools
pip install matplotlib

# For iree-samples and other benchmarking
-f https://github.com/iree-org/iree/releases
iree-compiler
iree-runtime
iree-tools-tflite
iree-tools-xla
iree-tools-tf

gin-config
tf-nightly
tf-models-nightly
tensorflow-text-nightly
transformers
jax[cpu]

Pillow

lit
pyyaml
```

## Path
- `./saved_model`: save the tf2 pretrained models. Now have 
	- [TF2] Imagenet (ILSVRC-2012-CLS) classification with MobileNet V2.)
- `/home/shenghao/dataset`: save other datasets, including:
	- `cifar-10-batches-py`
	- `adult.csv`
	- `ImageNet`

## TF 2.0 XLA with DLRM
Please refers to `./tensorflow_dlrm` for the DLRM model

