# 3DTeethSeg22_IGIP

# Getting started

## Training

First, install the python requirements by:

```shell
pip install -r requirements.txt
```

Next, some operators should be compiled in `code/pointnet2`. Run the following command to compile:

```shell
cd code/pointnet2/src
pip install -e .
```

You should train all the models in `code/models`. The training and testing config is written in `code/config/config.yml`. After adjusting the configuration files, just run the
python scripts in `code/models` directly to start training.

```shell
# Training steps
# Change any args you want
export CUDA_VISIBLE_DEVICES=0 && python code/models/teeth_gingival_seg.py
export CUDA_VISIBLE_DEVICES=0 && python code/models/centroids_prediction.py
export CUDA_VISIBLE_DEVICES=0 && python code/models/patch_segmentation.py
export CUDA_VISIBLE_DEVICES=0 && python code/models/teeth_classification.py
```

**You should better keep the default learning rate (lr) and epoch nums (n-epochs), they work good for us.** 

## Testing

There's no testing code in this repository. BUT, you can do inferences by `code/process.py`, this script is based on
the challenge's `process.py`.


# GPU and memory consumption

Here we show the GPU memory and host memory consumption when setting the specific batch size for reference.
The data is obtained on a machine with one NVIDIA RTX 3090. The batch size can be adjusted according to your device.

| Stage                     | Batch size | GPU memory | Host memory |
|---------------------------|------------|------------|-------------|
| teeth_gingival_separation | 16         | ~20GB      | ~4GB        |
| centroids_prediction      | 16         | ~16GB      | ~4GB        |
| patch_segmentaiton        | 16         | ~21GB      | ~5GB        |
| teeth_classification      | 64         | ~16GB      | ~5GB        |

# Suggested environment

The following environment is tested ok.

- Linux arch 5.15.74-1-lts
- GCC 9.3.0
- Python 3.8
- PyTorch 1.10.0
- CUDA 11.7


# Support

I do some code clean up based on my original codes, and I didn't re-train all the models. They should work fine, but 
there's no guarantee. If you have any problems, please e-mail me at [202135331@mail.sdu.edu.cn](mailto:202135331@mail.sdu.edu.cn).