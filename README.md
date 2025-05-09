# Image Analysis - Classification & Segmentation

This repository contains two machine learning models for image classification and segmentation tasks. It includes a **ResNet50-based U-Net model** for **cat vs dog classification** and a **U-Net model from scratch** for **lung image segmentation**.

## 📁 Folder Structure

```plaintext
project/
│
├── resnet50_unet/                   # ResNet50 model for cat vs dog classification
│   ├── main.py                      # Main script for training and testing ResNet50 U-Net
│   ├── model.py                     # ResNet50 U-Net model definition
│   ├── utils.py                     # Utility functions (data loading, preprocessing)
│   ├── test.py                      # Testing script for the ResNet50 model
│   └── visualization.py             # Visualization functions (plotting results)
│
├── u_net_from_scratch/              # U-Net model for lung image segmentation
│   ├── main.py                      # Main script for training and testing U-Net from scratch
│   ├── model.py                     # U-Net model definition from scratch
│   ├── utils.py                     # Utility functions for data processing
│   └── visualization.py             # Visualization functions for U-Net results
│
├── README.md                        # Project overview and instructions
├── requirements.txt                 # Python dependencies for the project
└── .gitignore                       # Files to be ignored by git
📝 Project Overview
This project implements two different deep learning models for image analysis tasks:

1. ResNet50 U-Net for Cat vs Dog Classification
A ResNet50-based U-Net model is used for the binary classification of images into two categories: cats and dogs. The model utilizes the pre-trained ResNet50 architecture as a backbone and adds U-Net-style skip connections to improve segmentation and classification performance.

2. U-Net from Scratch for Lung Image Segmentation
This model implements a U-Net architecture from scratch, designed for image segmentation of lung images. The goal is to identify and segment lung regions in medical images. U-Net is widely used for segmentation tasks due to its ability to preserve spatial information in images.
