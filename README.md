# WayFASTER: a Self-Supervised Traversability Prediction for Increased

![outline](images/WayFASTER.png)

## Introduction
The code and trained models of:

**WayFASTER: a Self-Supervised Traversability Prediction for Increased, [Mateus V. Gasparino](https://scholar.google.com/citations?user=UbtCA90AAAAJ&hl=en), [Arun N. Sivakumar](https://scholar.google.com/citations?user=peIOOn8AAAAJ&hl=en) and [Girish Chowdhary](https://scholar.google.com/citations?user=pf2zAXkAAAAJ&hl=en), ICRA 2024** [[Paper]]()

We presented WayFASTER, a novel method for self-supervised traversability estimation that uses sequential information to predict a map that improves the traversability map visibility. For such, we use a neural network model that takes a sequence of RGB and depth images as input, and uses the camera’s intrinsic and extrinsic parameters to project the information to a 3D space and predict a 2D traversability map.

## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@article{gasparino2024wayfaster,
  title={WayFASTER: a Self-Supervised Traversability Prediction for Increased Navigation Awareness},
  author={Gasparino, Mateus Valverde and Sivakumar, Arun Narenthiran and Chowdhary, Girish},
  journal={arXiv preprint arXiv:2402.00683},
  year={2024}
}
```

## System requirements
- Linux (Tested on Ubuntu 20.04)
- Python3 (Tested using Python 3.8) 
- PyTorch (Tested using Pytorch 1.13.1) 
- CUDA (Tested using CUDA 11.7)

## Installation
a. Create a python virtual environment and activate it.
```shell
python3 -m venv wayfaster
source wayfaster/bin/activate
```
b. Update `pip` to the latest version.
```shell
python3 -m pip install --upgrade pip
```
c. Install PyTorch 1.13.1 for CUDA 11.7.
```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
d. Install the other python dependencies using the provided `requirements.txt` file.
```shell
pip3 install -r requirements.txt
```

## WayFASTER dataset
- Download the WayFASTER dataset from [here](https://uofi.app.box.com/s/orehra8yt1xlh9mvv3yx9xe2776phtvx) and extract it to the `dataset` folder, outside of the `wayfaster` folder.
- In the `src/utils/train_config.py` file, update the `DATASET.TRAIN_DATA` and the `DATASET.VALID_DATA` parameter to the path of the dataset folder.

## Code execution
### Configuration parameters and training
The configuration parameters of the model such as the learning rate, batch size, and dataloader options are stored in the `src/utils` folder.
If you intend to modify the model parameters, please do so here. Also, the training and evaluation scripts are stored in the same folder.

### Model and data handling
The network model is stored in the `src/models` folder. The dataset analisys and handling are stored in the `scripts` folder.

To train the model, execute the following command. 
```shell
bash train_wayfaster.sh 
```

## Experimental results

![outline](images/waypoints.png)

As we can see in the figure above, WayFASTER was the most successful method among the three. The WayFAST approach wast not able to reach the final goal in most of the experiments, often failing when performing the final sharp turn. Our hypothesis is that, since the method could only use a narrow field-of-view, it was not able to respond in time when doing sharps turns, since obstacles would suddenly appear. LiDAR-based navigation and our method WayFASTER were able to successfully complete the whole path. This is because both these methods have wider field-of-view. However, WayFASTER was able to complete the path in less time, which is a good indication that it was able to predict the traversability map more accurately.

| Method       | Success   | Avg. time (s) |
|--------------|-----------|---------------|
| LiDAR-based  | 5/5       | 201           |
| WayFAST      | 1/5       | -             |
| WayFASTER    | 5/5       | 118           |

## License
This code is released under the [MIT](https://opensource.org/license/mit) for academic usage.
