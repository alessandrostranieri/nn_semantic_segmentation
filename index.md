---
title: "Welcome Page"
layout: default
---

<!--
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
      {{ post.excerpt }}
    </li>
  {% endfor %}
</ul>
-->

## Introduction

In this project we deal with the problem of semantic segmentation. Our aim is to devise a semantic segmentation architecture in order to leverage multiple data-sets. Before we delve into the description of how the project was carried out, let's provide some context. In the the nest sub-sections we will give an overview of what is semantic segmentation, some of the current result and we provide a motivation for the project.

The model, training and validation steps have been implemented using the [Keras](https://keras.io/) library. Other notable libraries include:

* numpy
* Pillow
* pandas
* matplotlib

## Semantic Segmentation

Semantic Segmentation denotes a class of computer vision problems, whose aim is to divide the region of an image into distinct
sub-areas. What this effectively translates to, is into labeling each individual pixel in the image, where the label carries a specific information.

In the type of information carried by the label lies the difference between two types of semantic segmentation: **pixel-wise** and **instance**.

In **pixel-wise** segmentation each pixel is assigned to a single class. This means that an algorithm will typically output an image of the same size, where each pixel value is the label corresponding to a class. For example 1 could be assigned to the class CAR and and 2 to the class PERSON.

In **instance** segmentation a position in the image will actually carry to pieces of information: The type of the segmented object and a single identifier for that object. This means
that if the image shows two cars, the pixel of the part of images will identify them as CAR 1 and CAR 2. If the algorithm worked well of course.

The following pictures are an example of an camera image, a real semantic segmentation image and a colorized semantic segmentation image.

|![alt text](images/example_original_kitti.png "Original")|![alt text](images/example_semantic_kitti.png "Original")|![alt text](images/example_semantic_rgb_kitti.png "Original")|
|:--:|:--:|:--:| 
| *Original Image* | *Semantic* | *Sematic RGB* |

Semantic segmentation can be considered as a step towards the general goal scene understanding. In image classification problems, the objective is to assign a label to an image, which state what is the pictured object. A more complicated problem can be classifying object and returning their bounding boxes. Semantic segmentation can be considered as a step further, where every pixel of the identified object is returned.

It becomes obvious then how this problem is of importance in applications such as autonomous driving or landscape monitoring. Semantic segmentation is also used in medical imaging, where it can be used to isolate brain region, growths or cell nuclei.

## Objective of the project

Producing annotated data-sets for semantic segmentation tasks is not easy. One consequence of that is that annotated data-sets can be quite limited in size. This naturally works against the large data-sets requirement of deep-learning algorithms. For this reason it would be an advantage to be able to leverage a combination of several data-sets. This of course comes with the complication that different data-sets are created with different image sizes and different labelling schemes.

## A necessary introduction: the datasets

After having understood the context of this project, the next necessary step consisted in getting acquainted with the data. For this project we gathered information about X data-sets, which we try to summarize in the following table. This is important because we need implement a way to feed images and from different sources to our model.

| Dataset    | URL                                                                                  | Size                                                                          | Info                                                                                                          |
|------------|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Cityscapes | [link](https://www.cityscapes-dataset.com/)                                          | 5000 Images: 3475 train/1525 test                                             | Images laid out by type, phase and city. Image names are different across types.                              |
| KITTI      | [link](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) | 200 Images in total                                                           |  Simple subdivision camera and semantic                                                                       |
| ADE20K     | [link](http://groups.csail.mit.edu/vision/datasets/ADE20K/)                          | 20.210 train images/2000 test images.                                         | The dataset includes many different types of images, so we must probably select a subset according to content |
| COCO       | [link](http://cocodataset.org/#home)                                                 | Detection(people): 200K images. 80 categories. Stuff(grass, wall) 55K images. | Needs API to extract masks.                                                                                   |

Different data-sets employ different labelling schemes. For simplicity, in this context we decided to work with only the **KITTI** and **Cityscapes** datasets, which use the same labelling scheme. The images provide labels for 33 (34 with 0-label segments) classes, listed in the following table.



### Data Generator

We are going to eventually have a model that we train in supervised fashion. That is we provide examples of input and out and the model should hopefully learn to produce accurate descriptions from new input data.

There are two ways in Keras to feed data during training and prediction: through the method `fit` or the method `fit_generator`. In the first method the data is expected to be already in a `numpy` array format, whereas with `fit_generator` one must provide a generator instance. In foresight of the need to combine batches of images from different sources, we opted to write a custom generator.

|![alt text](images/DataGenerator_01.png "Data Generator Diagram")|
|:--:| 
| *Caption* |

## Model

Our initial approach was to use an implementation of the DeepLab library. DeepLab is a deep architecture that makes use of CRFs and atrous convolution. WHAT ARE CRF and ATROUS. According to [2](#references) it provides the best accuracy on several image data-sets, compare to other architectures. Unfortunately, the model we tried to use did not show any sign of learning. In two epochs the model would reach the best, despite poor, loss value and stop improving on validation loss.

In order to continue we then opted for a much simpler architecture: U-Net. U-Net builds upon the *Fully Convolutional Network(FCN)*. This network was designed by t
It consists of a contracting path (left side) and an expansive path (right side). U-Nat was designed to work well will small data-sets. 

In particular we take inspiration from the code provided [here](https://github.com/zhixuhao/unet/blob/master/model.py). The model is fairly simple and it's easy to verify the correctness by simply following the network's diagram.

|![unet](images/U-Net_Image.png "U-Net Diagram")|
|:--:|
| U-Net diagram, as presented in the original paper [3](#references) |

## Combining the dataset

### Modification of data generator

### Loss functions

## Experiments

## Conclusions

## References

1. [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)
2. [A Review on Deep Learning Techniques Applied to Semantic Segmentation](https://arxiv.org/abs/1704.06857)
3. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

