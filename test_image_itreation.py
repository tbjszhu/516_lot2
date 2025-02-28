# this file is logically the same with test_image.py #
# please refer to the commentaire of that file       #

from sklearn.externals import joblib
import argparse
import cv2
from utils import *
from matplotlib import pyplot as plt
import time

desp_dim = 11
normalize_value = 255
sub_window = 32


def main(img_addr, model, hist_model, filter_enable, ed_enable):
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
        if prefix in filename and model in filename:
            print "Pixel kmeans model exists"
            model_exist = True
            model_addr = filename
            break
           
    if model_exist:
        print "Kmeans model addr : " + model_addr
        kmeans = joblib.load(model_addr)  # load pre-trained k-means model #
        # print ('kmeans parameters', kmeans.get_params())
    else:
        print "please generate kmeans model for pixel"
        sys.exit(0)

    # get k-means init value
    init_value = kmeans.get_params()['init']

    # train histogram
    test_dir = "./image_jpg/"
    desp_save_dir = "./descriptor/"
    hist_dir = "./hist/"
    image_list = getFileListFromDir(test_dir, filetype='jpg')
    hist_list = getFileListFromDir(hist_dir, filetype='npy')
    hist_num = len(hist_list)
    hist_total = []

    for image_file in image_list:
        test_image = image_file.split('/')[-1].split('.')[0] 
        print test_image
        # test_image = "nessne04"
        test_addr = test_dir + test_image + ".jpg"
        print "teating image : ", test_addr
        test_image = test_addr.split('/')[-1].split('.')[0]
        desp_list = getFileListFromDir(desp_save_dir, filetype='npy')
        desp_exist = False
        for despfile in desp_list:
            if test_image in despfile:
                desp_exist = True

        if desp_exist:
            print "reading exist descriptor file"
            des = np.load(desp_save_dir + test_image + "_dsp.npy")
        else:
            print "creating descriptor"
            img = cv2.imread(test_addr)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            des = descriptor_generator(img_lab)

        lines = des.shape[0] * des.shape[1]
        des_reshape = np.reshape(des, (lines, desp_dim))
        label = kmeans.predict(des_reshape)
        label_reshape = np.reshape(label, (des.shape[0], des.shape[1]))    
        pixel_kmeans = colorizeImage_16(label, label_reshape.shape)
        # plt.figure(1)
        # plt.imshow(pixel_kmeans.astype(np.uint8))


        # get 16 layer image
        layer_num = kmeans.get_params()['n_clusters']
        layers = np.zeros((label_reshape.shape[0], label_reshape.shape[1], layer_num))
        for layer in range(layer_num):
            tmp = label_reshape.copy()
            tmp[tmp == layer] = normalize_value
            tmp[tmp != 255] = 0
            '''plt.imshow(tmp.astype(np.uint8),cmap ="gray")
            plt.show()  
            plt.close()'''
            integral = generate_integral_image(tmp, normalize_value)
            '''plt.imshow(integral, cmap ="gray")
            plt.show()
            plt.close()'''
            layers[:, :, layer] = integral[0:-1, 0:-1].copy()

        # generate histogram
        hist_addr = hist_dir + test_image
        hist = generate_histogram(layers, sub_window)

        # debug
        '''hist_reshape = np.reshape(hist, (label_reshape.shape[0], label_reshape.shape[1],12))
        hist_pixel = np.zeros(label_reshape.shape)
        for i in range(label_reshape.shape[0]):
            for j in range(label_reshape.shape[1]):
                for k in range(12):
                    if hist_reshape[i,j,k] == np.max(hist_reshape[i,j,:]):
                        hist_pixel[i,j] = k
        plt.imshow(hist_pixel,cmap ="gray")
        plt.show()
        plt.close()'''

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
                print "Hist kmeans model exists"
                model_exist = True
                model_addr = filename
                if hist_model == "12":
                    if hist_model in filename: # "road" is the key word for searching model name
                        model_addr = filename
                        break
                else:                
                    if model == "12":
                        if "road" not in filename: # "road" is the key word for searching model name
                            model_addr = filename
                            break
                    elif model == "16":
                        if "road" in filename:
                            model_addr = filename
                            break
                    
        print "Kmeans model addr : " + model_addr
        if model_exist:
            kmeans_hist = joblib.load(model_addr)  # load pre-trained k-means model
        else:
            print "please generate kmeans model for histogram"
            sys.exit(0)

        # predict hist for test image
        print hist.shape
        label_hist = kmeans_hist.predict(hist)
        original = cv2.imread(test_dir + test_image + ".jpg")[1:-1, 1:-1, :]
        original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
        col_img = colorizeImage(original.shape,label_hist, hist_model)
        fus_img = fusionImage(original, original.shape,label_hist, model,hist_model, filter_enable, ed_enable)

        #label_hist_reshape = np.reshape(label_hist, label_reshape.shape)

        #show images
        """
        plt.figure(2)
        plt.imshow(original)
        plt.figure(3)
        plt.imshow(col_img)
        plt.figure(4)
        plt.imshow(fus_img)    
        plt.show()
        """
        f,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2)
        ax11.set_title("Original")
        ax11.imshow(original)
        ax11.set_axis_off()
        ax12.imshow(pixel_kmeans)
        ax12.set_title("Descriptor K-means output")
        ax12.set_axis_off()
        ax21.imshow(col_img)
        ax21.set_title("Histogram K-means output")
        ax21.set_axis_off()
        ax22.imshow(fus_img)
        ax22.set_title("Fusion output")
        ax22.set_axis_off()
        plt.tight_layout()
        #plt.show()
        
        rst_dir = "./image_result/"
        if os.path.exists(rst_dir) == False:
            os.makedirs(rst_dir)       
        
        plt.savefig(rst_dir + test_image + ".jpg")
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, default="helston3",
                        help="test image name")
    parser.add_argument("-d", type=str, default="16",
                        help="kmeans desp model version 12 or 16")
    parser.add_argument("-g", type=str, default="12",
                        help="kmeans hist model version 8 or 12")
    parser.add_argument("-f", type=int, default="1",
                        help="filtrage, 0 disable, other value enable ")
    parser.add_argument("-e", type=int, default="1",
                        help="eroded,dilate, 0 disable, other value enable")
                                                                                            
    args = parser.parse_args()
    img_addr = args.i
    model = args.d
    hist_model = args.g
    filter_enable = args.f
    ed_enable = args.e
    if hist_model == "12":
        model = "16"
    print "img_addr : %s" % (img_addr)
    main(img_addr, model, hist_model, filter_enable, ed_enable)
