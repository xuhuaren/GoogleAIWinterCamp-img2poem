#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
script

@author: liujy
"""

import os
import glob
from PIL import Image
import random as rd

datafile_path = "/data/256x256/"
output_path = "./data/sketch2image_set/"
A_domain = "sketch"
B_domain = "photo"
sub_folder_name ="tx_000000000000"

classnames = os.listdir(os.path.join(datafile_path, B_domain,sub_folder_name))


#generate output folder
if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(os.path.join(output_path,A_domain)):
    os.mkdir(os.path.join(output_path,A_domain))

if not os.path.exists(os.path.join(output_path,A_domain,"train")):
    os.mkdir(os.path.join(output_path,A_domain,"train"))
if not os.path.exists(os.path.join(output_path,A_domain,"val")):
    os.mkdir(os.path.join(output_path,A_domain,"val"))
if not os.path.exists(os.path.join(output_path,A_domain,"test")):
    os.mkdir(os.path.join(output_path,A_domain,"test"))
    
    
if not os.path.exists(os.path.join(output_path,B_domain)):
    os.mkdir(os.path.join(output_path,B_domain))    
    
if not os.path.exists(os.path.join(output_path,B_domain,"train")):
    os.mkdir(os.path.join(output_path,B_domain,"train"))
if not os.path.exists(os.path.join(output_path,B_domain,"val")):
    os.mkdir(os.path.join(output_path,B_domain,"val"))
if not os.path.exists(os.path.join(output_path,B_domain,"test")):
    os.mkdir(os.path.join(output_path,B_domain,"test"))    
    
for sub_class in classnames:
    
        
    if "." in sub_class :
        pass  
    
    else:
        imglist = os.listdir(os.path.join(datafile_path, B_domain,sub_folder_name, sub_class))
               
        for ix, sub_image in enumerate(imglist):
            
            if 'jpg' in sub_image:
                
                imgid = sub_image.split('.', 1)[0]
            
                # find corresponding sketch
                match_sketch = glob.glob(os.path.join(datafile_path, A_domain,sub_folder_name, sub_class,imgid+'*'))
                if match_sketch:
                    for iy, sub_matched in enumerate(match_sketch):  
                        
                        if rd.random <= 0.8:
                            subset = "train"
                        elif rd.random > 0.9:
                            subset = "test"
                        else:
                            subset = "val"                            
                            
                        im = Image.open(sub_matched)
                        im.convert('RGB').save(os.path.join(output_path,A_domain,subset, sub_class+"_"+imgid+"_"+str(iy)+'.jpg'),"JPEG")
                        im = Image.open(os.path.join(datafile_path, B_domain, sub_folder_name, sub_class, sub_image))
                        im.convert('RGB').save(os.path.join(output_path,B_domain,subset, sub_class+"_"+imgid+"_"+str(iy)+'.jpg'),"JPEG")                        
                        
                        
                        
