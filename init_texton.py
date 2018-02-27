import cv2
from utils import *
from matplotlib import pyplot as plt
from scipy.misc import imsave
import numpy as np 

dim = 3 # constant dimension for each pixel RGB or LAB

def local_texton_generation(addr, rectangle, texton_name, show = False):
    img = cv2.imread(addr)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    h_org = rectangle[0]
    w_org = rectangle[1]
    height = rectangle[2]
    weight = rectangle[3]

    img_lab_local = np.zeros((height, weight,dim))
    for i in range(height):
        for j in range(weight):                    
            img[h_org+i][w_org+j]=[0,0,255]
            for k in range(dim):
                img_lab_local[i][j][k] = img_lab[h_org+i][w_org+j][k]
    des = descriptor_generator(img_lab_local)
    texton = np.mean(np.mean(des, axis = 1), axis = 0) # get the descriptor average for this texton
    if show == True:
        plt.title(texton_name)
        plt.imshow(img)
        plt.show()
        plt.close()                
    return texton
     
def main():
    img_dir = "./image_jpg/"
    save_dir = "./descriptor_init/"
    generator_descriptor(img_dir, save_dir)
    rectangle = [0,0,0,0] # x start pos, y start pos, height, width for the texton example window
    
    image_list = getFileListFromDir(img_dir, filetype='jpg')
    show = False
    for addr in image_list:
        filename = addr.split('/')[-1].split('.')[0]
        
        # sky
        if filename == "nessne04":
            rectangle = [20,20,50,50]
            texton_name = 'sky'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)
        
        # grass    
        if filename == "nessne04":
            rectangle = [450,140,20,20]
            texton_name = 'grass'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)
            
        # tree    
        if filename == "nessne04":
            rectangle = [210,240,80,80]
            texton_name = 'tree'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)
            
        # yellow leaf on ground    
        if filename == "wolfrun":
            rectangle = [380,110,60,80]
            texton_name = 'leaf'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)            
            
        # water    
        if filename == "nessne04":
            rectangle = [318,596,40,40]
            texton_name = 'water'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)
            
        # cement road    
        if filename == "nessne04":
            rectangle = [270,190,200,100]
            texton_name = 'road'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)
            
        # sand road   
        if filename == "gilmore":
            rectangle = [280,170,180,350]
            texton_name = 'sand road'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)
            
        # oil road   
        if filename == "lsampford1":
            rectangle = [300,256,200,330]
            texton_name = 'oil road'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)

        # wet road   
        if filename == "follymill2":
            rectangle = [390,360,120,150]
            texton_name = 'wet road'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)

        # stone road   
        if filename == "wolfrun":
            rectangle = [330,270,140,170]
            texton_name = 'stone road'
            show = True
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            show = False            
            print texton
            np.save("./descriptor_init/"+texton_name, texton)                         
                              
        # car    
        if filename == "much4":
            rectangle = [250,60,80,100]
            texton_name = 'car'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)
            
        # house    
        if filename == "much4":
            rectangle = [100,440,80,100]
            texton_name = 'house'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)  
            
        # Route Sign    
        if filename == "runway":
            rectangle = [194,270,120,220]
            texton_name = 'sign'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)   
            
        # cloud    
        if filename == "m45_04sml":
            rectangle = [40,200,50,50]
            texton_name = 'cloud'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)
                                
        # animal    
        if filename == "gower02":
            rectangle = [280,320,60,60]
            texton_name = 'animal'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)  
            
        # fast moving leaf    
        if filename == "gower08":
            rectangle = [210,80,200,200]
            texton_name = 'fastleaf'
            texton = local_texton_generation(addr, rectangle, texton_name, show)
            print texton
            np.save("./descriptor_init/"+texton_name, texton)                                   

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
