import numpy as np
import cv2
import glob
import os
import sys
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from matplotlib import pyplot as plt

# Read images/masks from a directory
target_layers = 3 # number of road cluster 

def getFileListFromDir(img_dir, filetype='png'):
    """
    :param img_dir: imgs root dir, string
    :param filetype: "png", "jpg" or "bmp", string
    :return: list of images, list
    """
    img_dir = img_dir + '/*' + '*.' + filetype
    l = sorted(glob.glob(img_dir))  # ./merged_train/999/rotation/999_r.png
    return l


def getHistListFromDir(img_dir):
    """
    :param img_dir: imgs root dir, string
    :param filetype: "png", "jpg" or "bmp", string
    :return: list of images, list
    """
    filetype = 'npy'
    img_dir = img_dir + '/*.' + filetype
    l = sorted(glob.glob(img_dir))  # ./merged_train/999/rotation/999_r.png
    return l


# read images with img generator

def lab_generator(lab_list):
    """
    :param lab_list: list of iamges, list of string
    :return: yield a pair of sequences images
    """
    while len(lab_list) > 0:
        f1 = lab_list.pop(0)
        print "read file: ", f1.split('/')[-1]
        np_lab = np.load(f1)
        yield (np_lab, f1.split('/')[-1])


# read images with img generator

def img_generator(img_list):
    """
    :param img_list: list of iamges, list of string
    :return: yield a pair of sequences images
    """
    while len(img_list) > 0:
        f1 = img_list.pop(0)
        print "read img: ", f1.split('/')[-1]
        img = cv2.imread(f1, 0) # 0 is needed for brief discriptor
        yield (img, f1.split('/')[-1])


# calculate ORB descriptors
def descriptor_generator(data):
    """
    :param data: numpy array grayscale image for getting histo or local descriptors for an images
    :param feature_point_quantity: MAX feature point quantity
    :return: descriptor_list
    """
    height, width, dim =  data.shape
    mask_width = 3
    patch_width = (mask_width - 1)/2
    w1 = 0.5
    w2 = 1
    w3 = 0.5
    
    des = []
    for i in range(height):
        if i > patch_width - 1 and i < height - patch_width:
            des_line = []
            for j in range(width):                
                if j > patch_width-1 and j < width - patch_width:
                    des_pix = [0]*(mask_width*mask_width + 2)
                    Lc = data[i][j][0]
                    des_pix[0] = w1 * Lc
                    des_pix[1] = w2 * data[i][j][1]
                    des_pix[2] = w2 * data[i][j][2]
                    loop_time = mask_width * mask_width -1
                    index_move = 0
                    for m in range(mask_width):
                        for n in range(mask_width):
                            index = m*mask_width+n+3
                            if m==patch_width and n==patch_width:
                                index_move = -1
                                continue
                            Lc_t = data[i+m-patch_width][j+n-patch_width][0]
                            if Lc_t > Lc:
                                diff = Lc_t - Lc
                            else:
                                diff = Lc - Lc_t
                            des_pix[index+index_move] = w3 * diff
                    des_line.append(des_pix)
            des.append(des_line)                
    des = np.array(des)
    return des

# generator descriptors
def generator_descriptor(fileaddr, save_addr):
    """
    :param fileaddr: string, images dir
    :param save_addr: string, where to save
    :return: None
    """

    # create save directory
    if not os.path.isdir(save_addr):
        os.makedirs(save_addr)
        print 'create ' + save_addr
    if fileaddr[-1] != '/':
        fileaddr += '/'

    fileList = glob.glob(fileaddr + "*.png")
    #print fileList
    for file in fileList:
        if '.png' in file:
            print file
            filename_des = file.split('/')[-1].split('.')[0]
            data = cv2.imread(file)
            des = descriptor_generator(data)
            np.save(save_addr+'/'+filename_des+'_dsp', des)

def convert_BGR2LAB(addr):
    img = cv2.imread(addr)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img_lab
                
def read_kmeans_init(desp_init_dir):
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
    return init_value
            
def generate_kmeans_model(train_data, save_addr, n_clusters, ndarray=None):
    if ndarray == None:
        kmeans = KMeans(n_clusters, random_state=0).fit(train_data)
    else:
        pass
        kmeans = KMeans(n_clusters, init=ndarray, random_state=0).fit(train_data)  
    if os.path.exists(save_addr) == False:
        os.mkdir(save_addr)
    if cv2.__version__[0] == '3':
        prefix = 'cv3'
    else:
        prefix = 'cv2'
    # save k-means model for further use
    joblib.dump(kmeans, save_addr + '/' + prefix + '_kmeans_road_' + str(n_clusters) + '.pkl')

def generate_integral_image(tmp, normalize_value):
    tmp = 1.0*tmp/normalize_value
    integral = cv2.integral(tmp.astype(np.uint8))    
    #return np.sqrt(integral)
    return integral

def generate_histogram(layers, sub_window):
    """
    :param data: numpy array grayscale image for getting histo or local descriptors for an images
    :param feature_point_quantity: MAX feature point quantity
    :return: descriptor_list
    """
    
    height, width, dim =  layers.shape
    # a b
    # d c
    alloc_width = sub_window
    
    zeros = np.zeros((1,dim))
    hist = np.zeros_like(layers)

    for i in range(height):
        for j in range(width):
            if i < alloc_width or j < alloc_width: # treatment for edge element
                a = zeros
            else:
                a = layers[i-alloc_width,j-alloc_width] 
                                          
            if i < alloc_width:
                b = zeros
            else:
                b = layers[i-alloc_width,j]
                
            if j < alloc_width:
                d = zeros
            else:
                d = layers[i,j-alloc_width]

            c = layers[i, j]
            hist_pix = 1.0*c + a - d - b
            '''if i == 200 and j == 200:
                print "c", c
                print "a",a
                print "d",d
                print "b",b
                print i,j, np.max(hist_pix), hist_pix'''
            #hist.append(hist_pix)
            hist[i,j] = hist_pix  
    hist = np.reshape(hist, (height * width, dim))             
    return hist    

def colorizeImage(shape, label, hist_model):
    """
    :param shape: output image shape (h,w)
    :param label: kmeans hist prediction
    :return: RGB image of segmentation
    """
    # show segmentation result
    tmp = np.zeros((shape[0],shape[1],3))
    if hist_model == "8":
        map_label2color = [(200, 180, 0), (0, 200, 180), (180, 0, 200), (100, 0, 0), (0, 100, 0), (0, 0, 100),
                           (200, 100, 0), (100, 200, 0)]
    elif hist_model == "12":
        map_label2color = [(200, 180, 0), (0, 200, 180), (180, 0, 200), (100, 0, 0), (0, 100, 0), (0, 0, 100),
                           (200, 100, 0), (100, 200, 0),(100, 50, 0), (0, 100, 50), (50, 0, 100), (200, 100, 100)]    
    count = 0
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            tmp[i, j, :] = map_label2color[label[count]]
            count += 1
    return tmp
    
    
def colorizeImage_16(label, shape):
    """
    :param label: pixel kmeans prediction
    :return: RGB image of segmentation
    """
    # show segmentation result
    tmp = np.zeros((shape[0], shape[1], 3))
    map_label2color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 0, 0), (0, 100, 0), (0, 0, 100),
                       (200, 100, 0), (100, 200, 0), (255, 100, 0), (100, 255, 0), (0, 100, 255),(100, 50, 0), (0, 100, 50), (50, 0, 100), (200, 100, 100), (100, 200, 100)]
                       
    count = 0
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            map_label2color[label[count]]
            tmp[i, j, :] = map_label2color[label[count]]
            count += 1
    return tmp
    
def fusionImage(img, shape, label, model, hist_model, filter_enable, ed_enable):
    """
    :param shape: output image shape (h,w)
    :param label: kmeans hist prediction
    :return: RGB image of segmentation
    """
    # show segmentation result

    tmp = img.copy()
    if hist_model == "8":
        map_label2color = [(200, 180, 0), (0, 200, 180), (180, 0, 200), (100, 0, 0), (0, 100, 0), (0, 0, 100),
                           (200, 100, 0), (100, 200, 0)]
    elif hist_model == "12":
        map_label2color = [(200, 180, 0), (0, 200, 180), (180, 0, 200), (100, 0, 0), (0, 100, 0), (0, 0, 100),
                           (200, 100, 0), (100, 200, 0),(100, 50, 0), (0, 100, 50), (50, 0, 100), (200, 100, 100)] 
    count = 0

    if hist_model == "12" and filter_enable:
        label = filter(label, shape, ed_enable).copy()
            
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if model == "12" and hist_model == "8":
                if label[count] == 0 or label[count] == 6 or label[count] == 7:
                    tmp[i, j, :] = map_label2color[0]
            elif model == "16" and hist_model == "8":
                if label[count] == 0 or label[count] == 2 or label[count] == 7:
                    tmp[i, j, :] = map_label2color[0]
            elif hist_model == "12":
                if label[count] == 1 or label[count] == 2 or label[count] == 4:
                    tmp[i, j, :] = map_label2color[0]                   
            count += 1 
    return tmp
    
def filter(label, shape, ed_enable):

    layers = np.zeros((target_layers,shape[0],shape[1]))
    image_height = shape[0]
    height_thrs = image_height*0.6
    surface = [0]*target_layers
    surface_thrs = shape[0]*0.2*shape[1]*0.2
    height = [0]*target_layers 
    count = 0
    
    # divide label list in to cluster layer image
    for i in range(shape[0]):
        for j in range(shape[1]):    
            if label[count] == 1:
                layers[0,i,j] = 1
            elif label[count] == 2:
                layers[1,i,j] = 1
            elif label[count] == 4:
                layers[2,i,j] = 1
            count += 1 
            
    if ed_enable:        
        # image erode and dilate to delete noise region
        eroded_ratio = 0.06
        eroded_kernel_width = int(shape[0]*eroded_ratio)
        eroded_kernel_height = int(shape[1]*eroded_ratio)
        eroded_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(eroded_kernel_height,eroded_kernel_height))
        
        dilated_ratio = 0.10
        dilated_kernel_width = int(shape[0]*dilated_ratio)
        dilated_kernel_height = int(shape[1]*dilated_ratio)
        dilated_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilated_kernel_height,dilated_kernel_height))    

        for k in range(target_layers):
            plt.imshow(layers[k],cmap ="gray")
            plt.show()            
            eroded = cv2.erode(layers[k], eroded_kernel)
            plt.imshow(eroded,cmap ="gray")
            plt.show()              
            dilated = cv2.dilate(eroded, dilated_kernel)
            layers[k] = dilated
            plt.imshow(dilated,cmap ="gray")
            plt.show()     
    
    # calculate the position(height) of the hot zone        
    for k in range(target_layers):   
        for i in range(shape[0]):
            for j in range(shape[1]): 
                if layers[k,i,j] == 1:
                    surface[k] += 1
                    height[k] += i
        if surface[k]!= 0:            
            height[k] = height[k]/surface[k]        
    label = np.zeros_like(label)
    
    # reset label
    for k in range(target_layers):    
        if height[k] < height_thrs:# and surface[k] < surface_thrs:
            print "filtering layer: ", k
            layers[k,:,:] = 0
        count = 0 
        for i in range(shape[0]):
            for j in range(shape[1]):
                if layers[k,i,j] == 1:
                    if k == 0:
                        label[count] = 1
                    elif k == 1:
                        label[count] = 2                        
                    elif k == 2:
                        label[count] = 4        
                count += 1               
    return label        
          


  
