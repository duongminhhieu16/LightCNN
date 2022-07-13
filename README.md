# LightCNN
This model is borrowed from [A Light CNN for Deep Face Representation with Noisy Labels](https://arxiv.org/abs/1511.02683). The PyTorch version original code can be found [here](https://github.com/AlfredXiangWu/LightCNN). The official and original Caffe code can be found [here](https://github.com/AlfredXiangWu/face_verification_experiment).

## Installation
- Install [PyTorch](http://pytorch.org/) following the website.
- Clone this repository:
```Shell
git clone https://github.com/duongminhhieu16/LightCNN
```
- I currently run it on Python 3.9 with CUDA 11.4.

## Datasets
- Download MS-Celeb-1M clean list: [Google Drive](https://drive.google.com/file/d/0ByNaVHFekDPRbFg1YTNiMUxNYXc/view?usp=sharing)
- Or download VGG-Face2 dataset [Google Drive](https://drive.google.com/drive/folders/1pXrBWc_TisDqK7hEF2rOa4hGoeS-4A2E?usp=sharing)
- Install ```gdown``` to download datasets from Google Drive:
```Shell
pip install gdown
gdown <link-to-the-dataset>
```
## Training
- Prepare the dataset:
```Shell
python process_data.py
```
- After running ```process_data.py```, there will be two files: ```train_list.txt``` and ```test_list.txt```.
- Trainng from scratch or from checkpoint:
```Shell
python train.py
```
