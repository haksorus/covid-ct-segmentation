import torch
import numpy as np
import matplotlib.pyplot as plt

def predict(model, data, threshold, threshold_value):
    
    x = data["image"].unsqueeze(0).to("cuda") # Add batch dimension

    output = model(x) # Forward pass

    probs = torch.sigmoid(output)

    full_mask = probs.squeeze().cpu().detach().numpy()
    
    if threshold:
        res = full_mask > threshold_value
    else:
        res = full_mask

    return res

def visualize_mask(model, dataset, threshold = True, threshold_value = 0.5):
    
    fig, axes = plt.subplots(3, 3, figsize=(14,14)) 

    for i in range(3):

        num = int(np.random.uniform(0, len(dataset)))

        axes[i,0].imshow(dataset[num]["image"].numpy().squeeze(), cmap = 'bone')
        axes[i,0].set_title("Image")

        axes[i,1].imshow(dataset[num]["mask"].numpy().squeeze(), cmap = 'bone')
        axes[i,1].set_title("Ground Truth")

        axes[i,2].imshow(predict(model, dataset[num], threshold, threshold_value), cmap = 'bone')
        axes[i,2].set_title("Prediction (thresholded)")