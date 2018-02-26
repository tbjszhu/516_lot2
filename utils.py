import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
from itertools import groupby
import sys
from collections import OrderedDict
import random
import csv
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time

# Read images/masks from a directory
train_width = 10 # the number of train images for each class
opposite_image_num = 10 # the number of opposite images for each class

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
    print des.shape
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
        kmeans = KMeans(n_clusters, init=ndarray, random_state=0).fit(train_data)  
    if os.path.exists(save_addr) == False:
        os.mkdir(save_addr)
    if cv2.__version__[0] == '3':
        prefix = 'cv3'
    else:
        prefix = 'cv2'
    # save k-means model for further use
    joblib.dump(kmeans, save_addr + '/' + prefix + '_kmeans_' + str(n_clusters) + '.pkl')

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
    alloc_width = sub_window - 1
    
    zeros = np.zeros((1,dim))
    his = []
    for i in range(height):
        for j in range(width):
            if i < alloc_width or j < alloc_width: # treatment for edge element
                a = zeros           
                if i < alloc_width:
                    b = zeros
                else:
                    b = layers[i-alloc_width,j]
                    
                if j < alloc_width:
                    d = zeros
                else:
                    d = layers[i,j-alloc_width]
            else:
                a = layers[i-alloc_width,j-alloc_width]   
            c = layers[i,j]   
            his_pix = c - d - b + a
            his.append(his_pix)
    his = np.array(his)
    his = np.reshape(his,(his.shape[0], his.shape[2]))              
    return his    

# generator hists (global descriptors) from data (local descriptors or image)
def generateHist(model, data, data_type, nfeatures, decpt_type):
    """
    :param model: k-means model trained by descriptors in database.
    :param data: numpy array grayscale image for getting histo or local descriptors for an images
    :param data_type : string, "dscpt" or "image"
    :param decpt_type : string , type of descriptor, e.g. "orb"
    :return: an histogramme for the given data
    """

    if data_type == "dscpt":
        # from k-means prediction to histogram (global descriptor) (#bins = #classes)
        res = np.zeros((1, model.get_params()['n_clusters']), dtype=np.float32)
        if data is None:  # if orb cannot get any keypoints
            return res

        label = model.predict(data)
        for value in label:
            res[0, value] += 1.0

        return res / np.sum(res)  # normalized histogram

    elif data_type == "image":

        if decpt_type == "orb":
            kp, des = orb_descriptor_generator(data, nfeatures)

        elif decpt_type == "brief":
            kp, des = brief_descriptor_generator(data, nfeatures)

        elif decpt_type == "sift":
            kp, des = SIFT_descriptor_generator(data, nfeatures)
        else:
            print "Algo : " + decpt_type + " is not supported"

        # from k-means prediction to histogram (#bins = #classes)
        res = np.zeros((1, model.get_params()['n_clusters']), dtype=np.float32)
        if des is None:  # if orb cannot get any keypoints
            return res
        des = np.asarray(des, dtype=np.float32)
        #des_float = []
        #for i in range(len(des)):  # convert des from int to float to avoid type warning
        #    des_float.append(map(float, des[i]))
        label = model.predict(des)
        for value in label:
            res[0, value] += 1.0
        return res / np.sum(res)  # normalized histogram

    else:
        print ("data type error")
        return 0


# search similar images of a given target (gs image) in terms of distance of hists.

def searchFromBase(base_dir, target, model, nfeatures, descriptor_type, class_id = -1, has_hist = False):
    """
    :param base_dir: search base of images
    :param target: target image numpy grayscale image
    :param model : kmeans pretrained model
    :param has_hist : bool, if true base_dir is the addr for the hists, else it is the addr for images
    :return: a list of ranking of top 10 similars images, [ (index, distance value)] and a list of image dir
    """
    if has_hist:
        imgs_addr = getHistListFromDir(base_dir)
    else:
        imgs_addr = getImageListFromDir(base_dir)
    dist = {}
    target_hist = generateHist(model, target, 'image', nfeatures, descriptor_type).astype(np.float32)
    # print np.sum(target_hist)
    
    # calculate distance between target hist and base hists
    for idx, img_addr in enumerate(imgs_addr):
        img_gs = []
        hist = []
        if has_hist == False:
            #print img_addr
            img_gs = cv2.imread(img_addr, '0')
            hist = generateHist(model, img_gs, 'image', descriptor_type)
        else:
            hist = np.load(img_addr)
        dist[idx] = np.linalg.norm(hist - target_hist)  # eucudian distance

    # get the top rankings
    sorted_d = OrderedDict(sorted(dist.items(), key=lambda x: x[1]))
    dictlist = []
    for key, value in sorted_d.items():
        temp = [key, value] # [index, distance]
        class_actual = imgs_addr[key].split('/')[-1].split('_')[0]
        if class_actual == str(class_id):
            dictlist.append(temp)
            filename = imgs_addr[key].split('/')[-1]
            #print filename,value
    return dictlist, imgs_addr


def get_class_image_list(target_dir, class_name):
    if target_dir[-1] != '/':
        target_dir += '/'
    l = glob.glob(target_dir + class_name + '/*/*')
    return l


def generate_random_image_list(image_list, class_name, class_start, class_num, num):
    class_name = str(class_name)
    image_list_temp = image_list[:]  # to store different image classes
    same_class_image_count = 0
    same_class_list = []

    for item in image_list_temp:  # find the files from the same class
        if str(class_name) in item:
            same_class_list.append(item)
            same_class_image_count += 1
    for item in same_class_list:  # delete the files from the same class
        image_list_temp.remove(item)

    rand_file_num_list = []
    while True:  # generate "num" defferent file num
        rand_file_num = str(random.randint(class_start, class_start + (class_num - 1) * same_class_image_count - 1))
        if (rand_file_num not in rand_file_num_list):
            rand_file_num_list.append(rand_file_num)
        else:
            continue
        if len(rand_file_num_list) == num:
            break
    rand_image_list = []
    for item in rand_file_num_list:
        rand_image_list.append(image_list_temp[int(item) - class_start])    
    return rand_image_list, same_class_image_count

def csv_init(csv_file_path, kmeans, nfeatures, class_name, class_width, descriptor_type):
    csv_file_name = csv_file_path + '/kmeans_' + str(kmeans.get_params()['n_clusters']) + '_nf_' + str(nfeatures) + descriptor_type + '_class_' + class_name + '.csv'
    if os.path.exists(csv_file_path) == False:
        os.mkdir(csv_file_path)
    csvfile = file(csv_file_name, 'wb')
    index = range(class_width)
    index_str = map(str, index)
    file_header = ['id'] + index_str + ['Truth','Total']
    writer = csv.writer(csvfile)
    writer.writerow(file_header)
    return csvfile, writer


def csv_deinit(csvfile, writer, score_global):
    writer.writerow(['Conclusion'] + score_global[:-1] + ["-"] + [score_global[-1]])
    csvfile.close()    

def pr_image_generate(pr_list,descriptor_type,kmeans, nfeatures):
    """
    :param pr_list: a list of lists (filename, label, score)
    :param descriptor_type : string, descriptor type
    :param nfeatures: max number of keypoints
    :param kmeans : kmeans model
    :return: None
    """
    y_test =  [] # label
    y_score =  [] # prediction
    cls = pr_list[0][0].split('_')[0]
    for _, label, score in pr_list:
        y_test.append(label)
        y_score.append(score)

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    area = auc(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('class '+str(cls)+' '+str(descriptor_type)+'_k'+str(kmeans.get_params()['n_clusters'])+
                '_nf'+str(nfeatures)+' Precision-Recall AUC ={0:0.2f}'.format(
        area))
    if os.path.exists('./pr_figures') is False:
        os.mkdir('./pr_figures')
    plt.savefig('./pr_figures/'+"class_"+str(cls)+' '+str(descriptor_type)+' k'+str(kmeans.get_params()['n_clusters'])+
                ' nf'+str(nfeatures)+'.png')
    plt.close() # do not forget to close your figure after savefig, else the figure will overlap.
def pr_csv_generation(target_dir, sub_hist_addr, kmeans, nfeatures, descriptor_type, class_id = -1, has_hist=True):
    
    image_list = getImageListFromDir(target_dir)
    class_list = []
    dir_list = glob.glob(target_dir + '/*')
    for item in dir_list:
        class_list.append(item.split('/')[-1])
    class_num = len(class_list)
    class_start = int(class_list[0])

    if class_num <= 0:
        print "class_num 0 error"
        sys.exit(0)
    if class_id != -1:
        class_id = str(class_id)
        if class_id in class_list:
            class_list = [class_id]
        else:
            print "class_id: %d not in class_list" % (class_id)

    for class_name in class_list: # iteration for each class
        pr_list = []
        class_image_list = get_class_image_list(target_dir, class_name)
        random_image_list, class_width = generate_random_image_list(image_list, class_name, class_start, class_num, opposite_image_num)
        class_image_list.extend(random_image_list) # joint two lists together

        csv_file_path = './pr_csv'
        csvfile, writer = csv_init(csv_file_path, kmeans, nfeatures, class_name, train_width, descriptor_type)
        score_global = [0] * (train_width + 1)
        score_global_str = []
        Truth = 0        
        for target, target_filename in img_generator(class_image_list): # iteration for each test image from this class 
            target_filename = target_filename.split('.')[0] #todo documentation
            target_class = target_filename.split('_')[0]   #todo ducumentation

            score_vector = [0] * train_width
            score_total = 0
            if target_class == class_name:
                Truth = 1
            else:
                Truth = 0
            results, imgs_list = searchFromBase(sub_hist_addr, target, kmeans, nfeatures, descriptor_type, class_name, has_hist=True)
            count = 0
            for key, value in results:
                score = 1.0 / (value+0.001)
                filename = imgs_list[key].split('/')[-1]
                matched_class = filename.split('_')[0]
                if matched_class == class_name:
                    score_vector[count] = score
                    score_global[count] += score
                    score_total += score
                count += 1
            score_vector_str = map(str, score_vector)
            writer.writerow([str(target_filename)] + score_vector + [str(Truth), str(score_total)])
            pr_list.append([str(target_filename), Truth, score_total])  
            score_global[-1] += score_total
        score_global_str = map(str, score_global)
        csv_deinit(csvfile, writer, score_global_str)
        pr_image_generate(pr_list, descriptor_type, kmeans, nfeatures)
