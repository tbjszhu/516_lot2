from sklearn.externals import joblib
import argparse
import cv2
from utils import *

desp_dim = 11

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
    # get classisave_modelfication result for init value
    label = kmeans.predict(init_value)
    #print kmeans.cluster_centers_
    
    # test image
    test_addr = "./image_jpg/nessne04.jpg"
    img = cv2.imread(test_addr)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)        
    des = descriptor_generator(img_lab)
    lines = des.shape[0] * des.shape[1]
    des_reshape = np.reshape(des, (lines, desp_dim))
    label = kmeans.predict(des_reshape)
    print label.shape
    label_pixel = label * 10
    label_reshape = np.reshape(label_pixel, (des.shape[0], des.shape[1]))
    print label_reshape.shape
    cv2.imshow('mid', label_reshape.astype(np.uint8))
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()      
    
    
    
    
                
               
    
    
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
