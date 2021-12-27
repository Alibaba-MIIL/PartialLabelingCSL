[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-classification-with-partial/multi-label-classification-on-openimages-v6)](https://paperswithcode.com/sota/multi-label-classification-on-openimages-v6?p=multi-label-classification-with-partial)

# Multi-label Classification with Partial Annotations using Class-aware Selective Loss

<br> [Paper](https://arxiv.org/abs/2110.10955) |
[Pretrained models](https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/README.md#pretrained-models)

Official PyTorch Implementation

> Emanuel Ben-Baruch, Tal Ridnik, Itamar Friedman, Avi Ben-Cohen, Nadav Zamir, Asaf Noy, Lihi Zelnik-Manor<br/> DAMO Academy, Alibaba
> Group

**Abstract**

Large-scale multi-label classification datasets are commonly, and perhaps inevitably, partially annotated. That is, only a small subset of labels are annotated per sample.
Different methods for handling the missing labels induce different properties on the model and impact its accuracy.
In this work, we analyze the partial labeling problem, then propose a solution based on two key ideas. 
First, un-annotated labels should be treated selectively according to two probability quantities: the class distribution in the overall dataset and the specific label likelihood for a given data sample.
We propose to estimate the class distribution using a dedicated temporary model, and we show its improved efficiency over a naive estimation computed using the dataset's partial annotations.
Second, during the training of the target model, we emphasize the contribution of annotated labels over originally un-annotated labels by using a dedicated asymmetric loss.
Experiments conducted on three partially labeled datasets, OpenImages, LVIS, and simulated-COCO, demonstrate the effectiveness of our approach. Specifically, with our novel selective approach, we achieve state-of-the-art results on OpenImages dataset. 

<!-- ### Challenges in Partial Labeling
(a) In a partially labeled dataset, only a portion of the samples are annotated for a given class. (b) "Ignore" mode exploits only the annotated samples which may lead to a limited decision boundary. (c) "Negative" mode naively treats all un-annotated labels as negatives. It may produce suboptimal decision boundary as it adds noise of un-annotated positive labels. Also, annotated and un-annotated negative samples contribute similarly to the optimization. (d) Our approach aims at mitigating these drawbacks by predicting the probability of a label being present in the image.
 
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/intro_modes_CSL.png" align="center" width="1000" ></td>
  </tr>
</table>
</p> -->

## OpenImages 
We provide a direct and convenient access for the OpenImages (V6) dataset. This will enable common and reproduceble basline for benchmarking and future research.
See further details [here]

### Class-aware Selective Approach
An overview of our approach is summarized in the following figure:
 
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/CSL_approach.png" align="center" width="900" ></td>
  </tr>
</table>
</p>

### Loss Implementation 

Our loss consists of a selective approach that adjusts the training mode for each class individually and a partial asymmetric loss.
<!-- The selective approach is based on two probabilities quantities: label likelihood and label prior. The partial asymmetric loss emphasizes the contribution of the annotated labels.   -->
An implementation of the Class-aware Selective Loss (CSL) can be found [here](/src/loss_functions/partial_asymmetric_loss.py). 
- ```class PartialSelectiveLoss(nn.Module)```


## Pretrained Models
<!-- In this [link](MODEL_ZOO.md), we provide pre-trained models on various dataset.  -->
We provide models pretrained on the OpenImages dataset with different partial training-modes and architectures:

| Model | Architecture | Link | mAP |
| :---            | :---:      | :---:     | ---: |
| Ignore          | TResNet-M | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/CSL/opim_v6/mtresnet_opim_ignore.pth)      | 85.38       |
| Negative        | TResNet-M | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/CSL/opim_v6/mtresnet_opim_negative.pth)    | 85.85       |
| Selective (CSL) | TResNet-M  | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/CSL/opim_v6/mtresnet_opim_86.72.pth)      | 86.72       |
| Selective (CSL) | TResNet-L  | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/CSL/opim_v6/ltresnet_v2_opim_87.34.pth)   | **87.34**   |
 


## Inference Code (Demo)
We provide [inference code](infer.py), that demonstrates how to load the
model, pre-process an image and do inference. An example run of
OpenImages model (after downloading the relevant model):
```
python infer.py  \
--dataset_type=OpenImages \
--model_name=tresnet_m \
--model_path=./models_local/mtresnet_opim_86.72.pth \
--pic_path=./pics/10162266293_c7634cbda9_o.jpg \
--input_size=224
```

### Result Examples 
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/demo_examples.png" align="center" width="900" ></td>
  </tr>
</table>
</p>



## Training Code
Training code is provided in [train.py](train.py). Also, code for simulating partial annotation for the [MS-COCO dataset](https://cocodataset.org/#download) is available ([coco_simulation](src/helper_functions/coco_simulation.py)). In particular, two "partial" simulation schemes are implemented: fix-per-class(FPC) and random-per-sample (RPS).
- FPC: For each class, we randomly sample a fixed number of positive annotations and the same number of negative annotations.  The rest of the annotations are dropped.
- RPS: We omit each annotation with probability p.

Pretrained weights using the ImageNet-21k dataset can be found here: [link](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/MODEL_ZOO.md)\
Pretrained weights using the ImageNet-1k dataset can be found here: [link](https://github.com/Alibaba-MIIL/TResNet/blob/master/MODEL_ZOO.md)

Example of training with RPS simulation:
```
--data=/datasets/COCO/COCO_2014
--model-path=models/pretrain/mtresnet_21k
--gamma_pos=0
--gamma_neg=1
--gamma_unann=4
--simulate_partial_type=rps
--simulate_partial_param=0.5
--partial_loss_mode=selective
--likelihood_topk=5
--prior_threshold=0.5
--prior_path=./outputs/priors/prior_fpc_1000.csv
```

Example of training with FPC simulation:
```
--data=/mnt/datasets/COCO/COCO_2014
--model-path=models/pretrain/mtresnet_21k
--gamma_pos=0
--gamma_neg=3
--gamma_unann=4
--simulate_partial_type=fpc
--simulate_partial_param=1000
--partial_loss_mode=selective
--likelihood_topk=5
--prior_threshold=0.5
--prior_path=./outputs/priors/prior_fpc_1000.csv
```

### Typical Training Results

#### FPC (1,000) simulation scheme:
| Model                    | mAP         | 
| :---                     | :---:       |
| Ignore, CE               | 76.46       |
| Negative, CE             | 81.24       |
| Negative, ASL (4,1)      | 81.64       |
| CSL - Selective, P-ASL(4,3,1)  | **83.44**       |     


#### RPS (0.5) simulation scheme:
| Model                    | mAP        | 
| :---                     | :---:      |
| Ignore, CE               | 84.90      |
| Negative, CE             | 81.21      |
| Negative, ASL (4,1)      | 81.91      |
| CSL- Selective, P-ASL(4,1,1)  | **85.21**      |  


## Estimating the Class Distribution
The training code contains also the procedure for estimating the class distribution from the data. Our approach enables us to rank the classes based on predictions of a temporary model trained using the *Ignore* mode. [link](https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/cadc2afab73294a0e9e0799eec06b095e50e646e/src/loss_functions/partial_asymmetric_loss.py#L131)

#### Top 10 classes:
| Method                    | Top 10 ranked classes      | 
| :---                     | :---:      |
| Original                           | 'person', 'chair', 'car', 'dining table', 'cup', 'bottle', 'bowl', 'handbag', 'truck', 'backpack'   |
| Estiimate (Ignore mode)            | 'person', 'chair', 'handbag', 'cup', 'bench', 'bottle', 'backpack', 'car', 'cell phone', 'potted plant'      |
| Estimate (Negative mode)           | 'kite' 'truck' 'carrot' 'baseball glove' 'tennis racket' 'remote' 'cat' 'tie' 'horse' 'boat'    |


## Citation
```
@misc{benbaruch2021multilabel,
      title={Multi-label Classification with Partial Annotations using Class-aware Selective Loss}, 
      author={Emanuel Ben-Baruch and Tal Ridnik and Itamar Friedman and Avi Ben-Cohen and Nadav Zamir and Asaf Noy and Lihi Zelnik-Manor},
      year={2021},
      eprint={2110.10955},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
Several images from [OpenImages dataset](https://storage.googleapis.com/openimages/web/index.html) are used in this project.
Some components of this code implementation are adapted from the repository https://github.com/Alibaba-MIIL/ASL.
