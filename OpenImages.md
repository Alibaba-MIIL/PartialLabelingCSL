# Downloading OpenImages

The original OpenImages dataset is available [here](https://storage.googleapis.com/openimages/web/index.html). However, we found it might be challenging to download and process the dataset. 

Hence, we provide direct access for our processed OpenImages (V6) dataset.
This can enable a common baseline for benchmarking and reproducing the article results, and for future research.

#### Notice that when downloading and using OpenImages, one must comply with the original dataset license, which is provided [here](https://storage.googleapis.com/openimages/web/factsfigures.html#:~:text=red%20indicates%20negatives.-,Licenses,-The%20annotations%20are)

# Download

| Item                    |         | 
| :---                     | :---:      |
| Training data                    | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/public/Open_Images_V6/train.tar.gz)        | 
| Testing data           |  [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/public/Open_Images_V6/test.tar.gz)    |
| Annotationo file       | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/public/Open_Images_V6/data.csv)   |
| MID-format   | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/public/Open_Images_V6/mid_to_classes.pth)    |


### Data Pre-processing
Note that for reducing the dataset size, we resized all images such that the short edge is 256.


# Acknowledgment
OpenImages dataset official page: https://storage.googleapis.com/openimages/web/index.html<br>
The corresponding paper: https://arxiv.org/abs/1811.00982<br>
We thank the OpenImages team for curating this valuable dataset. 

<!---
Our motivation is to provide an easy and accessible way for downloading the dataset, and reproducing the article results.
!--->


