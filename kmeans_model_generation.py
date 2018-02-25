import cv2
from utils import *
from matplotlib import pyplot as plt
from scipy.misc import imsave 

desp_dim = 11

def main():
    img_dir = "./image_jpg/"
    image_list = getFileListFromDir(img_dir, filetype='jpg')
    image_num = len(image_list)
    
    lab_dir = "./image_lab/"
    lab_list = getFileListFromDir(lab_dir, filetype='npy')
    lab_num = len(lab_list)
    
    # convert image from RGB format to LAB fromat if necessary
    if lab_num != image_num:
        print "Converting to format LAB"
        for addr in image_list:
            #print addr
            filename = addr.split('/')[-1].split('.')[0]
            img = cv2.imread(addr)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            np.save("./image_lab/"+filename, img_lab)            
        lab_list = getFileListFromDir(lab_dir, filetype='npy')
        lab_num = len(lab_list)
    else:
        print "LAB image exists"

    desp_save_dir = "./descriptor/"
    desp_list = getFileListFromDir(desp_save_dir, filetype='npy')
    desp_num = len(desp_list)
    
    if desp_num != lab_num: # generate/update descriptor for each LAB image 
        print "Generating Descriptor image..."
        train_data = []       
        for np_lab,filename in lab_generator(lab_list):
            filename_des = filename.split('/')[-1].split('.')[0]
            #print filename_des
            des = descriptor_generator(np_lab)
            np.save(desp_save_dir+'/'+filename_des+'_dsp', des)
            
            # reshape descriptor from 3D to 2D
            des_quantity = des.shape[0] * des.shape[1]
            des_reshape = np.reshape(des_reshape, (des_quantity, desp_dim))
            
            if train_data == []:
                train_data = des_reshape
            else:                
                train_data = np.vstack((train_data, des_reshape))
                
        desp_list = getFileListFromDir(desp_save_dir, filetype='npy')
        desp_num = len(desp_list)
    else:                 # read descriptors and generate training data 
        print "Descriptor image exists"        
        print "Reading Descriptor image..."
        train_data = []
        for des, filename in lab_generator(desp_list):
            filename_des = filename.split('/')[-1].split('.')[0]
            #print filename_des
            
            # reshape des from 3D to 2D
            des_quantity = des.shape[0] * des.shape[1]
            des_reshape = np.reshape(des, (des_quantity, desp_dim))
            
            if train_data == []:
                train_data = des_reshape
            else:
                train_data = np.vstack((train_data, des_reshape))     
                                  
    # read k-means init value            
    desp_init_dir = "./descriptor_init/"
    desp_init_list = getFileListFromDir(desp_init_dir, filetype='npy')
    desp_init_num = len(desp_init_list)    
    
    if desp_init_num == 0:
        print "please generate init cluster center for kmeans"
        sys.exit(0)
        
    init_value = []
    for npyfile in desp_init_list:
        if init_value == []:
            init_value = np.load(npyfile)
        else:
            init_value = np.vstack((init_value, np.load(npyfile)))
     #print init_value
        
    # k-means model generation
    n_clusters = desp_init_num
    ndarray = init_value #(n_clusters, n_features)
    print "Kmeans shape : " + str(ndarray.shape)
    print "Kmeans tarin size : " + str(train_data.shape)
    generate_kmeans_model(save_addr, n_clusters, ndarray, train_data)       

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
