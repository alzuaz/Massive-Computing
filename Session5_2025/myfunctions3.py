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


    def edge_filter(self, row):
        global shared_space
        global shared_matrix

        r = row
        image = self.image
        my_filter = self.my_filter
        (rows, cols, depth) = image.shape

        # Create an empty row to store the result
        frow = np.zeros((cols, depth), dtype=np.float64)  # keep as float for precision

        # Go through each column in the row
        for i in range(cols):
            total = np.zeros(depth, dtype=np.float64)

            # h = horizontal shift
            for h in [-1, 0, 1]:
                x = i + h
                # Keep x inside image boundaries
                if x < 0:
                    x = 0
                elif x >= cols:
                    x = cols - 1

                # v = vertical shift
                for v in [-1, 0, 1]:
                    y = r + v
                    # Keep x inside image boundaries
                    if y < 0:
                        y = 0
                    elif y >= rows:
                        y = rows - 1

                    # Correct filter indices
                    r_f = v + 1
                    c_f = h + 1

                    total += image[y, x, :] * my_filter[r_f, c_f]

            frow[i, :] = total

        # Store the row into the shared result (casting only once)
        with shared_space.get_lock():
            shared_matrix[row, :, :] = frow.astype(np.uint8)

        return


# In[ ]:


#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

