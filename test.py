import cv2
from utils import *
from matplotlib import pyplot as plt
from scipy.misc import imsave 

def main():
    img_dir = "./image_jpg/"
    image_list = getImageListFromDir(img_dir, filetype='jpg')
    image_num = len(image_list)
    
    lab_dir = "./image_lab/"
    lab_list = getImageListFromDir(lab_dir, filetype='npy')
    lab_num = len(lab_list)
    
    # convert image from RGB format to LAB fromat if necessary
    if lab_num < image_num:
        print "Converting to format LAB"
        for addr in image_list:
            print addr
            filename = addr.split('/')[-1].split('.')[0]
            img = cv2.imread(addr)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            np.save("./image_lab/"+filename, img_lab)            
        lab_list = getImageListFromDir(lab_dir, filetype='npy')
        lab_num = len(lab_list)
    else:
        print "LAB image exists"

    desp_save_dir = "./descriptor/"
    desp_list = getImageListFromDir(desp_save_dir, filetype='jpg')
    desp_num = len(desp_list)
    
    # generate descriptor for each LAB image
    if desp_num < lab_num:        
        for np_lab,filename in lab_generator(lab_list):
            filename_des = filename.split('/')[-1].split('.')[0]
            print filename_des
            des = descriptor_generator(np_lab)
            np.save(desp_save_dir+'/'+filename_des+'_dsp', des)
    else:
        print "Descriptor image exists"            
            
    #             


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
