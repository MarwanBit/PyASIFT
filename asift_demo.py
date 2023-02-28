from asift import affine_detect, affine_skew
# Built-in modules
from multiprocessing.pool import ThreadPool     # Use multiprocessing to avoid GIL
import sys
import argparse
import os 
import csv

# Third party modules, opencv-contrib-python is needed
import cv2
import numpy as np

# Local modules
from utilities import Timer, log_keypoints, image_resize
from image_matching import init_feature, filter_matches, draw_match
from config import MAX_SIZE


#Now we will be writing the code which will take a Logo and perform ASIFT detection with the image

#compare images
def compare_asift_images(query_image_dir, data_image_dir, distance_threshold=20.0, match_threshold=10): 
    '''
    This function takes in two paths to different images and then returns the matches
    given some threshold set by the user in one of the other parameters.
    '''

    #Load images
    query_image = cv2.imread(query_image_dir, cv2.IMREAD_GRAYSCALE)
    data_image = cv2.imread(data_image_dir, cv2.IMREAD_GRAYSCALE)

    #initialize matcher
    detector, matcher = init_feature("sift-flann")

    #Make sure the images are the right size
    ratio_1 = 1
    ratio_2 = 1

    #Resizes images if necessary
    if query_image.shape[0] > MAX_SIZE or query_image.shape[1] > MAX_SIZE:
        ratio_1 = MAX_SIZE / query_image.shape[1]
        print("Large input detected, image 1 will be resized")
        img1 = image_resize(query_image, ratio_1)
    else:
        img1 = query_image

    if data_image.shape[0] > MAX_SIZE or data_image.shape[1] > MAX_SIZE:
        ratio_2 = MAX_SIZE / data_image.shape[1]
        print("Large input detected, image 2 will be resized")
        img2 = image_resize(data_image, ratio_2)
    else:
        img2 = data_image

    #Now let's extract the features from both images
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    query_keypoints, query_descriptors = affine_detect(detector, img1, pool=pool)
    data_keypoints, data_descriptors = affine_detect(detector, img2, pool=pool) 

    #Iterate through the list of matches
    # Another option is to use a distance metric (Euclidean Distance)
    # For each image select highest cosine similarity value with appropriate image
    #pairwise distance
    # use distance matrix and iterate through rows to get pairs 
    raw_matches = matcher.knnMatch(query_descriptors, trainDescriptors= data_descriptors, k=2)

    #return the list of matches
    return raw_matches


#first let's load some examples
logo_dir = "C:\\Users\\Marwa\\Downloads\\openlogo\\JPEGImages\\WilliamHill_sportslogo_7.jpg"
query_dir = "C:\\Users\\Marwa\\Downloads\\openlogo\\JPEGImages\\WilliamHill_sportslogo_72.jpg"


#get a folder with the logos, and create a CSV with a column corresponding to each image
def create_metrics_csv():
    with open("metrics.csv", "w") as metrics_file:
        writer = csv.writer(metrics_file, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        image_list_file = open("C:\\Users\\Marwa\\Downloads\\openlogo\\ImageSets\\Main\\all.txt", 'r')
        for image_file in image_list_file.readlines():
            writer.writerow(image_file)
        image_list_file.close()



#Now let's try asift on these two images
create_metrics_csv()