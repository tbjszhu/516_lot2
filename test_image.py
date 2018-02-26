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
        print "please generate kmeans model for pixel"
        sys.exit(0)
        
    # get k-means init value
    init_value = kmeans.get_params()['init']
    
    # train histogram
    test_dir = "./image_test/"
    desp_save_dir = "./descriptor/test/"
    hist_dir = "./hist/cv3/"
    image_list = getFileListFromDir(test_dir, filetype='npy')
    #hist_list = getFileListFromDir(hist_dir, filetype='npy')
    #hist_num = len(hist_list)
    #hist_total = []

    if not os.path.exists(desp_save_dir):
        os.makedirs(desp_save_dir)
    
    test_image = "a6024_070sml"
    test_addr = test_dir + test_image + ".jpg"
    desp_path = desp_save_dir+test_image+'_dsp.npy'
    print "treating image : ", test_addr
    test_image = test_addr.split('/')[-1].split('.')[0]

    desp_exist = False
    if os.path.exists(desp_path):
        desp_exist = True
            
    if desp_exist:
        print "reading exist descriptor file"     
        des = np.load(desp_path)
    else:
        print "creating descriptor"
        print test_addr
        img = cv2.imread(test_addr)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)        
        des = descriptor_generator(img_lab) #no padding , so the output shape != input shape
        np.save(desp_path, des)

    lines = des.shape[0] * des.shape[1]
    des_reshape = np.reshape(des, (lines, desp_dim))
    label = kmeans.predict(des_reshape)
    label_reshape = np.reshape(label, (des.shape[0], des.shape[1])) 
    #label_pixel = label_reshape
    #plt.figure(1)
    #plt.imshow(label_reshape.astype(np.uint8),cmap ="gray")
    #plt.show()
    
    # get 12 layer integral images
    layer_num = kmeans.get_params()['n_clusters']
    layers = np.zeros((label_reshape.shape[0], label_reshape.shape[1], layer_num))
    for layer in range(layer_num):
        tmp = label_reshape.copy()
        tmp[tmp==layer] = normalize_value
        tmp[tmp!=normalize_value] = 0
        integral = generate_integral_image(tmp, normalize_value)

        #plt.imshow(integral, cmap ="gray")
        #plt.show()
        #plt.close()
        layers[:,:,layer] = integral[0:-1, 0:-1].copy()
        
    # generate histogram        
    hist_addr = hist_dir + test_image
    his = generate_histogram(layers, sub_window)
               
    # read kmeans hist
    save_addr = './save_hist_model'
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
        kmeans_hist = joblib.load(model_addr) # load pre-trained k-means model
         #print ('kmeans parameters', kmeans.get_params())
    else:
        print "please generate kmeans model for hist"
        sys.exit(0)

    # predict hist for test image
    original = cv2.imread(test_addr)[1:-1, 1:-1, :]
    label_hist = kmeans_hist.predict(his)
    col_img = colorizeImage(original.shape,label_hist)

    #label_hist_reshape = np.reshape(label_hist, label_reshape.shape)

    #show images
    plt.figure(1)
    plt.imshow(original)
    plt.figure(2)
    plt.imshow(col_img)
    plt.show()
    plt.close()
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