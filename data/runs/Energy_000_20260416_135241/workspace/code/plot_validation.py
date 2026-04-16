import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

img1 = mpimg.imread('report/images/cs2_36_identification_enhanced2.png')
axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('CS2_36 Identification')

img2 = mpimg.imread('report/images/nasa_validation2.png')
axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('NASA Validation')

img3 = mpimg.imread('report/images/oxford_validation2.png')
axes[2].imshow(img3)
axes[2].axis('off')
axes[2].set_title('Oxford Validation')

plt.tight_layout()
plt.savefig('report/images/combined_validation.png')
