import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.filters import threshold_multiotsu
from skimage.io import imread

# Setting the font size for all plots.
matplotlib.rcParams['font.size'] = 9

# The input image.
image = imread('026c97_0.jpg', as_gray=True)

# Applying multi-Otsu threshold for the default value, generating
# three classes.
thresholds = threshold_multiotsu(image)

# Using the threshold values, we generate the three regions.
regions = np.digitize(image, bins=thresholds)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))

# Plotting the original image.
ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].axis('off')



# Plotting the Multi Otsu result.
ax[1].imshow(regions, cmap='jet')
ax[1].set_title('Multi-Otsu result')
ax[1].axis('off')

plt.subplots_adjust()

plt.savefig("processed_img.jpg")
