    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nicolina

Goal: apply varying color filters to image
    
Input:
    img_preprocess:    img that filter is applied to
    counter:           counter for respective color filter 
    
Output:
    mask:               mask of image depending on counter

"""

import cv2 as cv
import numpy as np

def image_preprocessing(img_preprocess, counter):
#    gamma = 0.6
#    lookUpTable = np.empty((1,256), np.uint8)
#    for i in range(256):
#        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
#    img_preprocess = cv.LUT(img, lookUpTable)
    

    # base filters
    hsv = cv.cvtColor(img_preprocess, cv.COLOR_BGR2HSV)
    if counter == 0:
        lower_brown = np.array([0, 60, 0])  #hue saturation value
        upper_brown = np.array([25, 255, 100])
        brown_mask = cv.inRange(hsv, lower_brown, upper_brown)
        
        lower_grey = np.array([80, 40, 37])  #hue saturation value
        upper_grey = np.array([180, 135, 124])
        grey_mask = cv.inRange(hsv, lower_grey, upper_grey)
      
        lower_silver = np.array([110, 10, 59])  #hue saturation value
        upper_silver = np.array([129, 92, 135])
        silver_mask = cv.inRange(hsv, lower_silver, upper_silver)
        mask = brown_mask + grey_mask + silver_mask
    
    elif counter == 1:
        lower_first = np.array([0, 52, 15])  #hue saturation value
        upper_first = np.array([180, 135, 90])
        first_mask = cv.inRange(hsv, lower_first, upper_first)
        mask = first_mask
    
    elif counter == 2:
        lower_second = np.array([0, 20, 95])  #hue saturation value
        upper_second  = np.array([180, 70, 176])
        second_mask = cv.inRange(hsv, lower_second, upper_second)
        
        lower_third = np.array([0, 14, 82])  #hue saturation value
        upper_third  = np.array([180, 54, 243])
        third_mask = cv.inRange(hsv, lower_third, upper_third)
        mask = second_mask + third_mask
        
    elif counter == 3:
        lower_second = np.array([0, 20, 95])  #hue saturation value
        upper_second  = np.array([180, 70, 176])
        second_mask = cv.inRange(hsv, lower_second, upper_second)

        lower_third = np.array([0, 14, 82])  #hue saturation value
        upper_third  = np.array([180, 54, 243])
        third_mask = cv.inRange(hsv, lower_third, upper_third)
        
        lower_fourth = np.array([0, 0, 81])  #hue saturation value
        upper_fourth  = np.array([180, 37, 191])
        fourth_mask = cv.inRange(hsv, lower_fourth, upper_fourth)
        mask =  second_mask + third_mask + fourth_mask 
        
    elif counter == 4:
        lower_fourth = np.array([0, 0, 81])  #hue saturation value
        upper_fourth  = np.array([180, 37, 191])
        fourth_mask = cv.inRange(hsv, lower_fourth, upper_fourth)
        mask = fourth_mask
        
    elif counter == 5:
        lower_fifth = np.array([0, 25, 10])  #hue saturation value
        upper_fifth  = np.array([180, 119, 190])
        fifth_mask = cv.inRange(hsv, lower_fifth, upper_fifth)
        mask = fifth_mask
        
    elif counter == 6:
        lower_sixth = np.array([0, 15, 137])  #hue saturation value
        upper_sixth  = np.array([180, 47, 239])
        sixth_mask = cv.inRange(hsv, lower_sixth, upper_sixth)
        
        lower_eighth = np.array([130, 20, 60])  #hue saturation value
        upper_eighth  = np.array([180, 55, 168])
        eighth_mask = cv.inRange(hsv, lower_eighth, upper_eighth)  

        mask = sixth_mask + eighth_mask
        
    elif counter == 7:
        lower_seventh = np.array([100, 34, 72])  #hue saturation value
        upper_seventh  = np.array([122, 95, 144])
        seventh_mask = cv.inRange(hsv, lower_seventh, upper_seventh)            
        mask = seventh_mask
    
    elif counter == 8:
        lower_eighth = np.array([130, 20, 60])  #hue saturation value
        upper_eighth  = np.array([180, 55, 168])
        eighth_mask = cv.inRange(hsv, lower_eighth, upper_eighth)  

        mask = eighth_mask
        
    elif counter == 9: # 1 und 2
        lower_first = np.array([0, 52, 15])  #hue saturation value
        upper_first = np.array([180, 135, 90])
        first_mask = cv.inRange(hsv, lower_first, upper_first)
        
        lower_second = np.array([0, 20, 95])  #hue saturation value
        upper_second  = np.array([180, 70, 176])
        second_mask = cv.inRange(hsv, lower_second, upper_second)
        
        mask = first_mask + second_mask
        
    elif counter == 10:
        lower_first = np.array([0, 52, 15])  #hue saturation value
        upper_first = np.array([180, 135, 90])
        first_mask = cv.inRange(hsv, lower_first, upper_first)
        
        lower_second = np.array([0, 20, 95])  #hue saturation value
        upper_second  = np.array([180, 70, 176])
        second_mask = cv.inRange(hsv, lower_second, upper_second)
        
        lower_third = np.array([0, 14, 82])  #hue saturation value
        upper_third  = np.array([180, 54, 243])
        third_mask = cv.inRange(hsv, lower_third, upper_third)
        mask = first_mask + second_mask + third_mask
        
    elif counter == 11:
        lower_sixth = np.array([0, 15, 137])  #hue saturation value
        upper_sixth  = np.array([180, 47, 239])
        sixth_mask = cv.inRange(hsv, lower_sixth, upper_sixth)
        
        lower_eighth = np.array([130, 20, 60])  #hue saturation value
        upper_eighth  = np.array([180, 55, 168])
        eighth_mask = cv.inRange(hsv, lower_eighth, upper_eighth)            
        mask = eighth_mask
        
        mask = sixth_mask + eighth_mask
        
#    result = cv.bitwise_and(img_preprocess, img_preprocess, mask = mask)  
    return mask