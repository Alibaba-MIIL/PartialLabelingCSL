# Multi-label Classification with Partial Annotations using Class-aware Selective Loss

<br> [Paper](https://arxiv.org/abs/2009.14119) |
[Pretrained models](MODEL_ZOO.md)

Official PyTorch Implementation

> Emanuel Ben-Baruch, Tal Ridnik, Avi Ben-Cohen, Nadav Zamir, Asaf Noy, Itamar
> Friedman, Lihi Zelnik-Manor<br/> DAMO Academy, Alibaba
> Group

**Abstract**

Large-scale multi-label classification datasets are commonly, and perhaps inevitably, partially annotated. That is, only a small subset of labels are annotated per sample.
Different methods for handling the missing labels induce different properties on the model and impact its accuracy.
In this work, we analyze the partial labeling problem, then propose a solution based on two key ideas. 
First, un-annotated labels should be treated selectively according to two probability quantities: the class distribution in the overall dataset and the specific label likelihood for a given data sample.
We propose to estimate the class distribution using a dedicated temporary model, and we show its improved efficiency over a naive estimation computed using the dataset's partial annotations.
Second, during the training of the target model, we emphasize the contribution of annotated labels over originally un-annotated labels by using a dedicated asymmetric loss.
Experiments conducted on three partially labeled datasets, OpenImages, LVIS, and simulated-COCO, demonstrate the effectiveness of our approach. Specifically, with our novel selective approach, we achieve state-of-the-art results on OpenImages dataset. Code will be made available.

### Challenges in Partial Labeling
 (a) In a partially labeled dataset, only a portion of the samples are annotated for a given class. (b) "Ignore" mode exploits only a subset of the samples which may lead to a limited decision boundary. (c) "Negative" mode naively treats all un-annotated labels as negatives. It may produce suboptimal decision boundary as it adds noise of un-annotated positive labels. Also, annotated and un-annotated negative samples contribute similarly to the optimization. (d) Our approach aims at mitigating these drawbacks by predicting the probability of a label being present in the image.
 
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/intro_modes_CSL.png" align="center" width="1000" ></td>
  </tr>
</table>
</p>

### Class-aware Selective Approach
An overview of our approach is depicted in the following figure.
 
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/CSL_approach.png" align="center" width="900" ></td>
  </tr>
</table>
</p>

## Training Code Now Available !

With great collaboration by [@GhostWnd](https://github.com/GhostWnd), we
now provide a [script](train.py) for fully reproducing the article
results, and finally a <u>modern</u> multi-label training code is
available for the community.


## Asymmetric Loss (ASL) Implementation
In this PyTorch [file](\src\loss_functions\losses.py), we provide 
implementations of our new loss function, ASL, that can serve as a
drop-in replacement for standard loss functions (Cross-Entropy and
Focal-Loss)

For the multi-label case (sigmoids), the two implementations are: 
- ```class AsymmetricLoss(nn.Module)```
- ```class AsymmetricLossOptimized(nn.Module)``` <br>

The two losses are bit-accurate. However, AsymmetricLossOptimized()
contains a more optimized (and complicated) way of implementing ASL,
which minimizes memory allocations, gpu uploading, and favors inplace
operations.

For the single-label case (softmax), the implementations is called:
- ```class ASLSingleLabel(nn.Module)```



## Pretrained Models
In this [link](MODEL_ZOO.md), we provide pre-trained models on various
dataset. 

## Validation Code
Thanks to external contribution of @hellbell, we now provide a
validation code that repdroduces the article results on MS-COCO:

```
python validate.py  \
--model_name=tresnet_l \
--model_path=./models_local/MS_COCO_TRresNet_L_448_86.6.pth
```

## Inference Code
We provide [inference code](infer.py), that demonstrate how to load our
model, pre-process an image and do actuall inference. Example run of
MS-COCO model (after downloading the relevant model):
```
python infer.py  \
--dataset_type=MS-COCO \
--model_name=tresnet_l \
--model_path=./models_local/MS_COCO_TRresNet_L_448_86.6.pth \
--pic_path=./pics/000000000885.jpg \
--input_size=448
```
which will result in:
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/example_inference.jpeg" align="center" width="600" ></td>
  </tr>
</table>
</p>

Example run of OpenImages model:
```
python infer.py  \
--dataset_type=OpenImages \
--model_name=tresnet_l \
--model_path=./models_local/Open_ImagesV6_TRresNet_L_448.pth \
--pic_path=./pics/000000000885.jpg \
--input_size=448
```
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/example_inference_open_images.jpeg" align="center" width="600" ></td>
  </tr>
</table>
</p>


## Citation
```
 @misc{benbaruch2020asymmetric, 
        title={Asymmetric Loss For Multi-Label Classification}, 
        author={Emanuel Ben-Baruch and Tal Ridnik and Nadav Zamir and Asaf Noy and Itamar Friedman and Matan Protter and Lihi Zelnik-Manor}, 
        year={2020}, 
        eprint={2009.14119},
        archivePrefix={arXiv}, 
        primaryClass={cs.CV} }
```

## Contact
Feel free to contact if there are any questions or issues - Emanuel
Ben-Baruch (emanuel.benbaruch@alibaba-inc.com) or Tal Ridnik (tal.ridnik@alibaba-inc.com).
