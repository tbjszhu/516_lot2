import cv2
from utils import *
from matplotlib import pyplot as plt
from scipy.misc import imsave 

desp_dim = 11 # dimension of the texton descriptor

def main():

    img_dir = "./image_jpg/"
    image_list = getFileListFromDir(img_dir, filetype='jpg')
    image_num = len(image_list)
    
    lab_dir = "./image_lab/"
    if os.path.exists(lab_dir) == False:
        os.makedirs(lab_dir) 
    lab_list = getFileListFromDir(lab_dir, filetype='npy')
    lab_num = len(lab_list)
   
    # convert image from RGB format to LAB fromat if necessary
    if lab_num != image_num:
        print "Converting to format LAB"
        for addr in image_list:
            #print addr
            filename = addr.split('/')[-1].split('.')[0]
            img_lab = convert_BGR2LAB(addr)
            np.save("./image_lab/"+filename, img_lab)            
        lab_list = getFileListFromDir(lab_dir, filetype='npy')
        lab_num = len(lab_list)
    else:
        print "LAB image exists"

    desp_save_dir = "./descriptor/"
    if os.path.exists(desp_save_dir) == False:
        os.makedirs(desp_save_dir) 
    desp_list = getFileListFromDir(desp_save_dir, filetype='npy')
    desp_num = len(desp_list)
     
    if desp_num != lab_num: 
    # generate/update descriptor for each LAB image    
        print "Generating Descriptor image..."
        train_data = []       
        for np_lab,filename in lab_generator(lab_list):
            filename_des = filename.split('/')[-1].split('.')[0]
            des = descriptor_generator(np_lab)
            np.save(desp_save_dir+'/'+filename_des+'_dsp', des)
            
            # reshape descriptor from 3D to 2D to adapt the kmeans input
            des_quantity = des.shape[0] * des.shape[1]
            des_reshape = np.reshape(des, (des_quantity, desp_dim))
            
            if train_data == []:
                train_data = des_reshape
            else:                
                train_data = np.vstack((train_data, des_reshape))
                
        desp_list = getFileListFromDir(desp_save_dir, filetype='npy')
        desp_num = len(desp_list)
    else:                 *
        # read descriptors and generate training data 
        print "Descriptor image exists"        
        print "Reading Descriptor image..."
        train_data = []
        for des, filename in lab_generator(desp_list):
            filename_des = filename.split('/')[-1].split('.')[0]
            
            # reshape descriptor from 3D to 2D to adapt the kmeans input
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


    # k-means model generation
    save_addr = './save_model/'
    n_clusters = desp_init_num
    ndarray = init_value 
    # the form of the ndarray is "n_clusters * n_features"
    print "Kmeans shape : " + str(ndarray.shape)
    print "Kmeans tarin size : " + str(train_data.shape)
    generate_kmeans_model(train_data, save_addr, n_clusters, ndarray)       

if __name__ == "__main__":
    main()
