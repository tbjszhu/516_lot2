1. init_texton.py

This script is to choose initial texton from the images in the data set. For example, teston for road, tree or grass, etc. The texton descriptors are then stored in npy file for further use.

2. pixel_kmeans_model.py

This script is to generate a Kmeans model for the classification of the pixel(as described in the paper Bias). The init value of Kmeans is the init texton descriptor calculated in the init_texton.py. Including calor space translation, pixel descriptor generation and training for the Kmeans.

3. histogram_kmeans_model.py

This script is to generate a Kmeans model for the classification of the histogram. Including the calculation of the intergral image, image layer generation, histogram calculation and traning for the Kmeans of the histogram.

4. test_image.py

This script is to predict the road position from the input image. Including kmeans pixel classification, kmeans histogram generation and classification, cluster fusion, noise filter, etc. 

run example: python test_image.py -d 16 -g 12 -f 1 -e 1 -i helston4

5. test_image_iteration.py

Just a iteration for all the image in the test set. Logically the same with test_image.py.

6. utils.py

Tool functions to be imported in the above scripts.

