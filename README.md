Differentiable Histogram Loss Functions for Intensity-based Image-to-Image Translation
-------------------------------------------------------------
by Mor Avi-Aharon, Assaf Arbelle and Tammy Riklin Raviv


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


