# Covid-19 CT Segmentation UNet-baseline (lungs + infection)

## About work

**This Project includes 2 Task :**

1. Covid 19 Infection Segmentation
2. Covid 19 Lungs Segmentation

**Project pipeline:**

<p align="center">
  <img src="https://user-images.githubusercontent.com/69139386/180862045-86a8006c-7515-432c-a6f5-f62cb62a3674.png" />
</p>

# Preprocessing Stage

**1. As mentioned earlier, slices are first processed with CLAHE (Contrast Limiting Adaptive Histogram Equalization)**


<p align="center">
  <img src="https://user-images.githubusercontent.com/69139386/180862534-b012b870-a3e8-430f-b59f-57bd90acdced.png" />
</p> 


<p align="center">
  The impact of the filter on the sharpness of the image is clearly identifiable.
</p> 

**2. After CLAHE, I've cropped the ROI:**

<p align="center">
  <img src="https://user-images.githubusercontent.com/69139386/180863203-275adf4c-90e3-402a-adea-db676c7080a3.png" />
</p> 

**3. The next step is removing incomplete and fauty images (empty masks, e.g.)**

# Training stage

* As a model I've used the baseline **UNet**.
* For training subsets I've used the **simple transforms (Horizontal/Vertical flips)** from Albumentations
* I've used Catalyst **runner** for train (it's very comfortable to use) with **wandb** logger
* **30 epochs** + combination of losses (**DiceLoss, IouLoss, BCELoss**) + **Adam** optimizer + **CosineAnnealingWarmRestarts** scheduler for training loop

# Results

For baseline we have: 

1) Task 1 (Covid 19 Infection Segmentation) : 
* **IoU = 0.60**
* **DICE = 0.75**
2) Task 2 (Covid 19 Lungs Segmentation) : 
* **IoU = 0.71** 
* **DICE = 0.83**
