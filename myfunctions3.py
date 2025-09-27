#!/usr/bin/env python
# coding: utf-8

# # My Functions Library
# 
#
import numpy as np


# In[ ]:




from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes



# In[ ]:


#This functions just create a numpy array structure of type unsigned int8, with the memory used by our global r/w shared memory
def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

#this function creates the instance of Value data type and initialize it to 0
def dot_init(g_A):
    global A 
    A = g_A #We create a variable of type "double"           
    
    
def shared_dot_1(V):
    #This code is wrong!!!
    for f in V:
        A.value += f[0]*f[1]
    
def shared_dot_2(V):
    #This code is wrong!!!
    with A.get_lock():
        for f in V:
            A.value += f[0]*f[1]
    
def shared_dot_3(V):
    #This code is wrong!!!
    a=0
    for f in V:
        a += f[0]*f[1]
    with A.get_lock():
        A.value += a
    

def init_sharedarray(shared_array,img_shape):
    global shared_space
    global shared_matrix

    shared_space = shared_array
    shared_matrix = tonumpyarray(shared_space).reshape(img_shape)

#This function initialize the global shared memory data
class filter:
    
    def __init__(self,srcimg, imgfilter):
        #shared_array_: is the shared read/write data, with lock. It is a vector (because the shared memory should be allocated as a vector
        #srcimg: is the original image
        #imgfilter is the filter which will be applied to the image and stor the results in the shared memory array
        
        #We defines the local process memory reference for shared memory space
        #Assign the shared memory  to the local reference
        
        #Here, we will define the readonly memory data as global (the scope of this global variables is the local module)
        self.image = srcimg
        self.my_filter = imgfilter
        
        #here, we initialize the global read only memory data
        self.size = srcimg.shape
        
        #Defines the numpy matrix reference to handle data, which will uses the shared memory buffer
        


# In[ ]:


        
#this function just copy the original image to the global r/w shared  memory 
    def parallel_shared_imagecopy(self,row):

        global shared_space
        global shared_matrix
        
        image = self.image
        my_filter = self.my_filter
        # with this instruction we lock the shared memory space, avoiding other parallel processes tries to write on it
        with shared_space.get_lock():
            #while we are in this code block no ones, except this execution thread, can write in the shared memory
            shared_matrix[row,:,:]=image[row,:,:]
        return


# In[ ]:


    def edge_filter(self,row):

        global shared_space
        global shared_matrix
        
        r = row
        image = self.image
        my_filter = self.my_filter
        (rows,cols,depth) = image.shape
        #fetch the r row from the original image
        srow=image[row,:,:]
        if ( row>0 ):
            prow=image[row-1,:,:]
        else:
            prow=image[row,:,:]

        if ( row == (rows-1)):
            nrow=image[row,:,:]
        else:
            nrow=image[row+1,:,:]

        #defines the result vector, and set the initial value to 0

        #frow = srow
        #while we are in this code block no ones, except this execution thread, can write in the shared memory
        #shared_matrix[row,:,:]=frow

        frow = np.zeros((cols,depth))
        #frow=np.zeros((cols,depth))
        for i in range(cols):
            total = np.zeros(depth)
            r_f = 1
            c_f = 1 # value of original element

            # h = horizontal shift
            for h in [-1, 0, 1]:
                # recalculate the x pos
                x = i + h #column postion
                c_f += h #col position in filter
                if h == -1:
                    x = max(x, 0)
                if h == 1:
                    x = min(x, cols-1)
                # v = vertical shift
                for v in [-1, 0, 1]:

                    # recalculate the y pos
                    y = r + v #row position
                    r_f += v #row position in filter
                    if v == -1:
                        y = max(y, 0)
                    if v == 1:
                        y = min(rows-1, y)

                    '''
                    # case where both 'x' and 'y' are within the bounds of the image
                    if 0 <= y < cols and 0 <= x < rows:
                        total += image[y, x, :]

                    else:
                    '''

                    total += image[y, x, :]*my_filter[r_f, c_f]
                    
            frow[i,:] = total.astype(np.uint8)
        
        
        with shared_space.get_lock():
            frow= frow.astype(np.uint8)
            shared_matrix[row, :,:] = frow
        return

# In[ ]:


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

