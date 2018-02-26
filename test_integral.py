import cv2
import numpy as np
from matplotlib import pyplot as plt

a = np.ones((5,5))
print a
c = cv2.integral(a)[0]
plt.imshow(c.astype(np.uint8), cmap='gray')
plt.show()