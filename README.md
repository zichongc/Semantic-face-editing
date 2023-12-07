# StyleGAN-based Semantic Face Editing

This implementation is based on [InterfaceGAN (CVPR 2020)](https://genforce.github.io/interfacegan/). 
In lieu of utilizing StyleGANv1 within InterfaceGAN, I have customized the codebase to align with StyleGANv2. 
This adaptation ensures compatibility and leverages the enhanced features offered by StyleGANv2.

## Getting start

Download the model checkpoints from [here](https://drive.google.com/drive/folders/10zWfpPSj3EKzbhsg-VBexgiW24wb1PtJ?usp=sharing). Place them to `./checkpoints`.

Model checkpoints include `e4e_ffhq_encode.pt`, `shape_predictor_68_face_landmarks.dat`, `stylegan2-ffhq-config-f.pt`.

### Requirements
I have tested on:
* torch (PyTorch 1.13.0)
* torchvision (0.14.0)
* CUDA 11.6

## Steps for preparing attribute vector

### Train an attribute classifier
* Train an attribute classifier based on your specified attributes.
* Place the pretrained classifier to `./checkpoints/classifiers` and name it `[ATTRIBUTUE].pth`.

### Data preparation
* Prepare the data for attribute vector solving.

```shell
python preparing.py --attr=[ATTRIBUTE] --n_samples=20000
```

### Solve attribute vector

```shell
python solve.py --attr=[ATTRIBUTE] --code=w
```
* The solved attribute vectors will be saved to `./checkpoints/attribute_vectors`.

_Adjust the placeholders such as [ATTRIBUTE] according to your specific attribute names._

## Testing on generated images
Single attribute manipulation only.

```shell
python manipulation.py --attr=[ATTRIBUTE]
```

Results will be saved to `./outputs`.


## Inference

Provide a facial image for semantic editing. Make sure checkpoints are well-prepared. 
```shell
python inference.py --input=[IMAGE_PATH] --attr=[ATTRIBUTE] --alpha=3 --conditions=[ATTRIBURES(optinal)]
```
Results will be saved to `./outputs`.

## Acknowledgements
The StyleGANv2 is borrowed from this [pytorch implementation](https://github.com/rosinality/stylegan2-pytorch) by [@rosinality](https://github.com/rosinality).
The implementation of e4e projection is also heavily from [encoder4editing](https://github.com/omertov/encoder4editing).