# Reduced-Cost Hyperspectral Convolutional Neural Networks

## Description

Hyperspectral imaging provides a useful tool for extracting complex information when visual spectral bands are not enough to solve certain tasks. However, processing hyperspectral images is usually computationally expensive due to the great amount of both spatial and spectral data they incorporate.
Here, we present a low-cost convolutional neural network called Hyper3DNet designed for hyperspectral image classification. Its architecture consists of two parts: a series of densely connected 3-D convolutions used as a feature extractor, and a series of 2-D separable convolutions used as a spatial encoder.

## Network architecture

<img src=https://github.com/GiorgioMorales/hyper3dnet/blob/master/Network.png alt="alt text" width=700 height=400>
Fig: Hyper3DNet architecture with a 25x25x30x1 HSI input. The network includes a 3-D feature extractor (left) and a 2-D spatial encoder (right).

## Datasets

We used three well-known remote sensing HSI datasets: Indian Pines (IP), Pavia University (PU), and Salinas (SA). 

We also experimented with the EuroSAT dataset (EU) in spite of the fact that it is not a hyperspectral but a multispectral dataset, 
so that we validate the usefulness of our network for images with just a few spectral channels. The original EUROSAT dataset can be downloaded from 
[here](https://github.com/phelber/EuroSAT). Alternatively, a pre-processed ready-to-use dataset that combines all the images in a single ".h5" file can be downloaded from
[here](https://montana.box.com/s/wqakb91vp3fwe272ctx88n791s4gnqvj).

Furthermore, we use an in-greenhouse controlled HSI dataset of Kochia leaves in order to classify three different herbicide-resistance levels (herbicide-susceptible, dicamba-resistant, and glyphosate-resistant). 
A total of 76 images of kochia with varying spatial resolution and 300 spectral bands ranging from 387.12 to 1023.5 nm were captured. From these images, which were previously calibrated and converted to reflectance values, we manually extracted 6,316 25x25 pixel overlapping patches. The Kochia dataset can be downloaded from [here](https://montana.box.com/s/mhpi7mxlw68abb616v0zl9t03zfwue63).

## Summary of results

### Computational Efficiency

<img src=https://github.com/GiorgioMorales/hyper3dnet/blob/master/results/table1.png alt="alt text" width=500 height=170>

### Classification results: Indian Pines, Pavia University, and Salinas Datasets

<img src=https://github.com/GiorgioMorales/hyper3dnet/blob/master/results/table2.png alt="alt text" width=480 height=150>

### Classification results: EuroSAT Dataset

<img src=https://github.com/GiorgioMorales/hyper3dnet/blob/master/results/table3.png alt="alt text" width=550 height=210>

### Classification results: Kochia Dataset

<img src=https://github.com/GiorgioMorales/hyper3dnet/blob/master/results/table4.png alt="alt text" width=480 height=150>

## Usage

Python requirements can be installed from requirements.txt:

`pip install -r requirements.txt`

This repository contains the following scripts:

* `TrainCrossval.py`: Load data and trained the selected network using 10-fold cross-validation. The weights are saved in the \weigths folder.
* `CalculateMetrics-Plot.py`: Load data, calculate the performance metrics of the selected network, and plot segmentation results. The weights are laoded from the \weigths folder.
* `TestDemo.ipynb`: Jupyter file that evaluates the performance metrics of a selected network.
* `network.py`: Contains all the network architectures used in this work.
* `utils.py`: Additional methods used to transform the data and calculate metrics.
