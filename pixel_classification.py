from sklearn.externals import joblib
import argparse
import cv2
from utils import *
from matplotlib import pyplot as plt
import time

desp_dim = 11
normalize_value = 255
sub_window = 32

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
        kmeans = joblib.load(model_addr) # load pre-trained k-means model #
         #print ('kmeans parameters', kmeans.get_params())
    else:
        print "please generate kmeans model"
        sys.exit(0)
        
    # get k-means init value
    init_value = kmeans.get_params()['init']
    
    # train histogram
    test_dir = "./image_jpg/"   
    desp_save_dir = "./descriptor/"
    hist_dir = "./hist/"
    image_list = getFileListFromDir(desp_save_dir, filetype='npy') # change to test_dir
    hist_list = getFileListFromDir(hist_dir, filetype='npy')
    hist_num = len(hist_list)
    hist_total = []
    
    #!! change to !=
    if hist_num < len(image_list): # if you want to update hists, delete all of them
        for test_addr in image_list:
            #test_image = "nessne04"
            #test_addr = test_dir + test_image + ".jpg"
            print "teating image : ", test_addr
            test_image = test_addr.split('/')[-1].split('.')[0]
            desp_list = getFileListFromDir(desp_save_dir, filetype='npy')
            desp_exit = False
            for despfile in desp_list:
                if test_image in despfile:
                    desp_exist = True
                    
            if desp_exist:
                print "reading exist descriptor file"     
                des = np.load(desp_save_dir + test_image + ".npy")
            else:
                print "creating descriptor"
                img = cv2.imread(test_addr)
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)        
                des = descriptor_generator(img_lab)
                
            lines = des.shape[0] * des.shape[1]
            des_reshape = np.reshape(des, (lines, desp_dim))
            label = kmeans.predict(des_reshape)
            label_reshape = np.reshape(label, (des.shape[0], des.shape[1])) 
            label_pixel = label_reshape * 15
            #plt.imshow(label_reshape.astype(np.uint8),cmap ="gray")
            #plt.show()
            
            # get all layer image
            layer_num = kmeans.get_params()['n_clusters']
            layers = np.zeros((label_reshape.shape[0], label_reshape.shape[1], layer_num))
            for layer in range(layer_num):
                tmp = label_reshape.copy()
                tmp[tmp==layer] = normalize_value
                tmp[tmp!=255] = 0
                integral = generate_integral_image(tmp, normalize_value)
                '''plt.imshow(integral, cmap ="gray")
                plt.show()
                plt.close()'''
                layers[:,:,layer] = integral[0:-1,0:-1].copy()
                
            # generate histogram        
            hist_addr = hist_dir + test_image            
            his = generate_histogram(layers,sub_window)
            if os.path.exists(hist_dir) == False:
                os.mkdir(hist_dir)
            np.save(hist_addr, his)
            if hist_total == []:
                hist_total = his
            else:
                hist_total = np.vstack([hist_total, his])
    else:
        print "Hist exist, Reading..."
        for hist_addr in hist_list:
            print "Read file:", hist_addr
            his = np.load(hist_addr)
            if hist_total == []:
                hist_total = his
            else:
                hist_total = np.vstack([hist_total, his])                
    # kmeans
    print hist_total.shape
    save_addr = './save_hist_model/'
    n_clusters = 8
    generate_kmeans_model(hist_total, save_addr, n_clusters)
    
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
