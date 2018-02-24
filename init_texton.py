import cv2
from utils import *
from matplotlib import pyplot as plt
from scipy.misc import imsave 

def main():
    img_dir = "./image_lab/"
    save_dir = "./descriptor/"
    generator_descriptor(img_dir, save_dir)


if __name__ == "__main__":
    '''parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50,
                        help="Number of feature point for each image.")
    parser.add_argument("-c", type=int, default=25,
                        help="Number of cluster for kmeans")
    parser.add_argument("-d", type=str, default='orb',
                        help="Descriptor Type")                                               
    parser.add_argument("--addr", type=str, default='./min_merged_train/',
                        help="training set addr")                        

    args = parser.parse_args()
    
    train_addr = args.addr # './min_merged_train/' # path where train images lie
    desptype= args.d #'orb'  # type of descriptors to be generated
    nfeatures = args.n # 200 # Max quantity of kp, 0 as invalid for brief
    n_clusters = args.c # 200 # Max quantity of kp, 0 as invalid for brief
    print "train_addr : %s, desptype : %s, nfeatures : %d, nclusters : %d " % (train_addr, desptype, nfeatures, n_clusters)'''
    main()
