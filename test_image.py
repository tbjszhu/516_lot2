from sklearn.externals import joblib
import argparse
import cv2
from utils import *
from matplotlib import pyplot as plt

desp_dim = 11 # dimension of the texton descriptor
normalize_value = 255 # default pixel value for layer inlier
sub_window = 32 # window size to calculate histogram 

def main(test_image, model, hist_model, filter_enable, ed_enable):

    # decide whether a pixel model exist
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
            print "Kmeans model addr : " + model_addr
            break
            
    if model_exist:
        # load pre-trained pixel k-means model
        kmeans = joblib.load(model_addr)  
    else:
        print "please generate kmeans model for pixel"
        sys.exit(0)

 
    test_dir = "./image_test/"
    desp_save_dir = "./descriptor/"
    hist_dir = "./hist/"
    hist_list = getFileListFromDir(hist_dir, filetype='npy')
    hist_num = len(hist_list)
    hist_total = []

    test_addr = test_dir + test_image + ".jpg"
    print "teating image : ", test_addr
    test_image = test_addr.split('/')[-1].split('.')[0]
    
    # read/generate the descriptor file of the images  
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
        
    # Kmeans classification for each pixel
    lines = des.shape[0] * des.shape[1]
    des_reshape = np.reshape(des, (lines, desp_dim))
    label = kmeans.predict(des_reshape)
    label_reshape = np.reshape(label, (des.shape[0], des.shape[1]))    
    pixel_kmeans = colorizeImage_16(label, label_reshape.shape)

    # get 16 layer image
    layer_num = kmeans.get_params()['n_clusters']
    layers = np.zeros((label_reshape.shape[0], label_reshape.shape[1], layer_num))
    for layer in range(layer_num):
        tmp = label_reshape.copy()
        tmp[tmp == layer] = normalize_value
        tmp[tmp != 255] = 0
        integral = generate_integral_image(tmp, normalize_value)
        layers[:, :, layer] = integral[0:-1, 0:-1].copy()

    # generate histogram
    hist_addr = hist_dir + test_image
    hist = generate_histogram(layers, sub_window)

    # read histogram kmeans model
    save_addr = './save_hist_model'
    model_dir = ''
    model_exist = False
    model_list = getFileListFromDir(save_addr, filetype='pkl')
    if cv2.__version__[0] == '3':
        prefix = 'cv3'
    else:
        prefix = 'cv2'
    for filename in model_list: # choose the kmeans model to use
        if prefix in filename:
            print "Hist kmeans model exists"
            model_exist = True
            model_addr = filename
            if hist_model == "12":
                if hist_model in filename: # "road" is the key word for searching model name
                    model_addr = filename
                    break
            elif hist_model == "8" and hist_model in filename:                
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

    # Kmeans classification for histogram
    print hist.shape
    label_hist = kmeans_hist.predict(hist)
    original = cv2.imread(test_dir + test_image + ".jpg")[1:-1, 1:-1, :]
    original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
    col_img = colorizeImage(original.shape,label_hist, hist_model)
    fus_img = fusionImage(original, original.shape,label_hist, model,hist_model, filter_enable, ed_enable)

    #show road detection result in 2x2 figures
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
    plt.show()
    plt.close()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, default="helston4",
                        help="test image name")
    parser.add_argument("-d", type=str, default="16",
                        help="kmeans descriptor model dimension 12 or 16")
    parser.add_argument("-g", type=str, default="12",
                        help="kmeans histogram model quantity 8 or 12")
    parser.add_argument("-f", type=int, default="1",
                        help="filtrage, 0 disable, other value enable ")
    parser.add_argument("-e", type=int, default="1",
                        help="eroded&dilate, 0 disable, other value enable")
                                                                                            
    args = parser.parse_args()
    img_addr = args.i
    model = args.d
    hist_model = args.g
    filter_enable = args.f
    ed_enable = args.e
    
    # 12D histogram need the descriptor dimension to be 12 
    if hist_model == "12":
        model = "16"
    print "img_addr : %s" % (img_addr)
    main(img_addr, model, hist_model, filter_enable, ed_enable)
