from sklearn.externals import joblib
import argparse
import cv2
from utils import *
from matplotlib import pyplot as plt
import time

desp_dim = 11 # dimension of the texton descriptor
normalize_value = 255 # default pixel value for layer inlier
sub_window = 32 # window size to calculate histogram 

def main():

    # decide whether a model exist
    save_addr = './save_model'
    model_dir = ''
    model_exist = False
    model_list = getFileListFromDir(save_addr, filetype='pkl')
    if cv2.__version__[0] == '3':
        prefix = 'cv3'
    else:
        prefix = 'cv2'
    for filename in model_list:
        if prefix in filename:
            print "kmeans model exists"
            model_exist = True
            model_addr = filename 
    print "Kmeans model addr : " + model_addr     
    if model_exist:
        # load pre-trained k-means model
        kmeans = joblib.load(model_addr) 
    else:
        print "please generate kmeans model"
        sys.exit(0)
    
    # train histogram
    test_dir = "./image_jpg/"   
    desp_save_dir = "./descriptor/"
    hist_dir = "./hist/cv3/"
    desp_list = getFileListFromDir(desp_save_dir, filetype='npy') # change to test_dir
    hist_list = getFileListFromDir(hist_dir, filetype='npy')
    hist_num = len(hist_list)
    hist_total = []
    
    desp_exit = False
    if hist_num < len(desp_list): # if hist need to be update, delete all of them from the repo
        # generate new histogram 
        for test_addr in desp_list:
            print "treating image : ", tesnormalize_valuet_addr
            test_image = test_addr.split('/')[-1].split('.')[0]
            for despfile in desp_list:
                if test_image in despfile:
                    desp_exist = True
                    
            if desp_exist:
                print "reading exist descriptor file", despfile    
                des = np.load(desp_save_dir + test_image + ".npy")
            else:
                print "creating descriptor"
                img = cv2.imread(test_addr)
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)        
                des = descriptor_generator(img_lab)
                
            # Kmeans classification for each pixel    
            lines = des.shape[0] * des.shape[1]
            des_reshape = np.reshape(des, (lines, desp_dim))
            label = kmeans.predict(des_reshape)
            label_reshape = np.reshape(label, (des.shape[0], des.shape[1])) 
            
            # divide org image into layer images
            layer_num = kmeans.get_params()['n_clusters']
            layers = np.zeros((label_reshape.shape[0], label_reshape.shape[1], layer_num))
            for layer in range(layer_num):
                tmp = label_reshape.copy()
                tmp[tmp==layer] = normalize_value
                tmp[tmp!=255] = 0
                integral = generate_integral_image(tmp, normalize_value)
                layers[:,:,layer] = integral[0:-1,0:-1].copy()
                
            # generate histogram        
            hist_addr = hist_dir + test_image            
            his = generate_histogram(layers,sub_window)
            if os.path.exists(hist_dir) == False:
                os.makedirs(hist_dir)
            np.save(hist_addr, his)
            if hist_total == []:
                hist_total = his
            else:
                hist_total = np.vstack([hist_total, his])
    else:
        # read exist histogram
        print "Hist exist, Reading..."
        for hist_addr in hist_list:
            print "Read file:", hist_addr
            his = np.load(hist_addr)
            print his.shape
            if hist_total == []:
                hist_total = his
            else:
                hist_total = np.vstack([hist_total, his])
    # kmeans model generation
    save_addr = './save_hist_model/'
    n_clusters = 12 # 8
    generate_kmeans_model(hist_total, save_addr, n_clusters)
    
if __name__ == "__main__":
    main()
