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

In this project we deal with the problem of semantic segmentation. Our aim is to devise modify a semantic segmentation
architecture in order to leverage multiple data-sets. Before we delve into the description of how the project was carried out, let's provide some context.
In the the nest sub-sections we will give an overview of what is semantic segmentation, some of the current result and we provide a motivation for the project.

## Semantic Segmentation

Semantic Segmentation denotes a class of computer vision problems, whose aim is to divide the region of an image into distinct
sub-areas. What this effectively translates to, is into labeling each individual pixel in the image, where the label carries a specific information.

In the type of information carried by the label, lies the difference between two types of semantic segmentation: **pixel-wise** and **instance**.

In **pixel-wise** segmentation each pixel is assigned to a single class. This means that an algorithm will typically output an image of the same size, where each pixel value is the label corresponding to a class. For example 1 could be assigned to the class CAR and and 2 to the class PERSON.

In **instance** segmentation a position in the image will actually carry to pieces of information: The type of the segmented object and a single identifier for that object. This means
that if the image shows two cars, the pixel of the part of images will identify them as CAR 1 and CAR 2. If the algorithm worked well of course. :)

## How is it used?

## Objective of the project

Manually labelling a data-set for semantic segmentation is relatively cumbersome. One consequence of that is that annotated data-sets can be quite limited in size. As we know, in order for
a deep learning method to work, we need large amount of data. For this reason it would be an advantage to be able to leverage a combination of several data-sets. This of course comes with the complication that different data-sets are created with different image sizes and different labelling schemes.

## A necessary introduction: the datasets

Apart from understanding the context of this project, the very first step consisted in getting acquainted with the data. For this project we gathered information about X data-sets, which we try to summarize in the following table. This is important because we need implement a way to feed images and from different sources to our model. Typically a semantic segmentation dataset comes with two main sets of data:

* The original images, for example an RGB camera picture
* A label image, typically in gray scale

The label images are typically not very informative to the human eye


| Dataset        | URL | Size | Info |
|----------------|-----|------|------|
| Cityscapes     | 0   | 0    | 0    |
| Kitti          | 0   | 0    | 0    |
| ADE20K         | 0   | 0    | 0    |
| COCO           | 0   | 0    | 0    |
| Multi-Spectral | 0   | 0    | 0    |
| ApolloScape    | 0   | 0    | 0    |

### Data Generator

We are going to eventually have a model that we train in supervised fashion. That is we provide examples of input and out and the model should hopefully learn to produce accurate descriptions from new input data.

There are two ways in Keras to feed data during training and prediction: through the method `fit` or the method `fit_generator`. In the first method the data is expected to be already in a `numpy` array format, whereas with `fit_generator` one must provide a generator instance. In foresight of the need to combine batches of images from different sources, we opted to write a custom generator.

![alt text](/images/DataGenerator_01.png "Data Generator Diagram")

## Model

## Experiments

## Conclusions