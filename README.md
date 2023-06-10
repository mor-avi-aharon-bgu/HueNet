<div align="center">

# Differentiable Histogram Loss Functions for Intensity-based Image-to-Image Translation ([TPAMI 2023](https://ieeexplore.ieee.org/document/10133915))<br><sub><sub>Official TensorFlow Implementation</sub></sub>
</div> 

## Abstract
We introduce the HueNet - a novel deep learning framework for a differentiable construction of intensity (1D) and joint (2D) histograms and present its applicability to paired and unpaired image-to-image translation problems. The key idea is an innovative technique for augmenting a generative neural network by histogram layers appended to the image generator. These histogram layers allow us to define two new histogram-based loss functions for constraining the structural appearance of the synthesized output image and its color distribution. Specifically, the color similarity loss is defined by the Earth Mover's Distance between the intensity histograms of the network output and a color reference image. The structural similarity loss is determined by the mutual information between the output and a content reference image based on their joint histogram. Although the HueNet can be applied to a variety of image-to-image translation problems, we chose to demonstrate its strength on the tasks of color transfer, exemplar-based image colorization, and edges → photo, where the colors of the output image are predefined. 

Prerequisites
-------------------------------------------------------------
The code runs on linux machines with NVIDIA GPUs. 


Installation 
-------------------------------------------------------------
- Tensorflow 2.0 `pip install tensorflow-gpu`
- Tensorflow Addons `pip install tensorflow-addons`
- (if you meet "tf.summary.histogram fails with TypeError" `pip install --upgrade tb-nightly`)
- scikit-image, oyaml, tqdm
- Python 3.6
	
- For pip users, please type the command:
	pip install -r requirements.txt


DeepHist edges2photos
-------------------------------------------------------------
Download a edges2photos dataset (e.g edges2shoes)

`python ./datasets/download_pix2pix_datasets.py`

* edit `dataset_name` in `./datasets/download_pix2pix_datasets.py`  for other dataset

Train a model:
`python train.py --dataroot /home/<user_name>/.keras/datasets --task edges2photos`


DeepHist colorization
-------------------------------------------------------------
Download a CycleGAN dataset (e.g. summer2winter_yosemite):

`bash ./datasets/download_cyclegan_datasets.sh summer2winter_yosemite`

Train a model:
`python train.py --dataroot ./datasets --task colorization`


DeepHist color transfer
-------------------------------------------------------------
Download and unzip 102 Flower Category Database from:
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

Train a model:
`python train.py --dataroot ./datasets --task color_transfer`


References
-------------------------------------------------------------
```sh
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

Citation
-------------------------------------------------------------
If you find either the code or the paper useful for your research, cite our paper:
```sh
@ARTICLE{10133915,
  author={Avi-Aharon, Mor and Arbelle, Assaf and Raviv, Tammy Riklin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Differentiable Histogram Loss Functions for Intensity-based Image-to-Image Translation}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPAMI.2023.3278287}}
```
