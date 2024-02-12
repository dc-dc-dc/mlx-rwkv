# mlx-rwkv  

⚠️WIP⚠️ Implementation of [rwkv](https://arxiv.org/abs/2305.13048) in [mlx](https://github.com/ml-explore/mlx).

## Installation

Currenlty only depends on mlx and torch(to load the weights and convert to safetensors)

```shell
pip install -r ./requirements.txt
```

### Convert the weights to safetensors

Download weights from huggingface, currently this repo was developed with [rwkv5-world-1b](https://huggingface.co/RWKV/rwkv-5-world-1b5)

```shell
python convert.py ./rwkv5-world-1b.pth
```

This will take a while and create a file `rwkv5-world-1b.safetensors` that could be used by the model.
