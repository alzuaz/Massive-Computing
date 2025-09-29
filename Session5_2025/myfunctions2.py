#!/usr/bin/env python
# coding: utf-8

# # My Functions Library
# 
# Due a bug in the multiprocessing module implemented for Windows Operating System, the functions which will be executed in the parallel threads MUST be implemented in a separated file, and import them in the main programs.
# 
# In order to be loaded in you own program, you have to write your own functions here, and export to a python ".py" file, to be imported in  the main script.
# 
# To export to a python file, select in the *File*  menu, the option *Download as* and save as *Python .py* file.

# ## Functions needed for FirstParallel notebook

import numpy as np


class Matrix_Mult:
    M_1 = None
    M_2 = None

    def __init__(self, M1,M2):
        self.M_1 = M1
        self.M_2 = M2
        
    def cell_calcule(self,t):
        # t is the tuple with the cell coordinate (row,column)
        r,c=t
        _,z = self.M_1.shape
        accu = 0
        for i in range(z):
            accu += self.M_1[r,i]*self.M_2[i,c]
        return accu

    def row_calcule(self,idx):
        # v: is the input row
        # matrix_2: is the second matrix, shared by memory
        
        #here we calculate the shape of the second matrix, to generate the resultant row
        
        (rows,columns)=self.M_2.shape
        
        #we allocate the final vector of size the number of columns of matrix_2
        d=np.zeros(columns)
        
        #we calculate the dot product between vector v and each column of matrix_2
        for i in range(columns):
            accu = 0
            for r in range(rows):
                accu += self.M_1[idx,r]*self.M_2[r,i]
            d[i]=accu
        
        #returns the final vector d
        return d

# In[ ]:

class Filter:
    filter_mask = None
    image = None
    def __init__(self,filter_mask):
        self.filter_mask = filter_mask

    def init_globalimage(self,img):
        self.image=img

    def parallel_filtering_image(self,r):
        # r: is the image row to filter
        # image is the global memory array
        # filter_mask is the filter shape to apply to the image

        #This a lazzy code. To avoid to write all the time self.image, we assign to local variable image.
        
        image = self.image
        #from the global variaable, gets the image shape
        (rows,cols,depth) = image.shape
    
        #fetch the r row from the original image
        srow=image[r,:,:]
        #Here, we analize if we are in the row0 o last row of the image 
        if ( r>0 ):
            prow=image[r-1,:,:]
        else:
            prow=image[r,:,:]
        
        if ( r == (rows-1)):
            nrow=image[r,:,:]
        else:
            nrow=image[r+1,:,:]
        
        #defines the result vector, and set the initial value to 0
        frow=np.zeros((cols,depth))

        ########################
        ########################
        # HERE YO HAVE TO WRITE WOUR CODE TO CALCULATE EACH PIXEL IN THE r IMAGE ROW!!! 
        
        ## YOU HAVE TO REPLACE THE FOLLOW CODE
        frow=srow #THIS CODE IS NOT VALID, JUST AN EXAMPLE
        
        #return the filtered row
        return frow
    





# In[ ]:


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

