
# Leeds Butterfly Dataset
# Performing SVD on Butterfly image dataset which orginally consists of 832 images of mostly 256*256 dimensions.
# Attaching 5 images for trial of code

import numpy as np
from random import randrange
import os
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg

# Here, picking up a random image from the dataset file in C drive,converting it to grayscale and showing original image
# Mode L is normally interpreted as grayscale.

filenames = os.listdir('C:/dataset')
img = Image.open('C:/dataset/' + filenames[randrange(len(filenames))]).convert('L').resize((256,256))
plt.imshow(img,cmap='gray')
plt.show()

def svd(A, tol=1e-5):
    #tol=1e-5 signifies the tolerance. The tolerance is for excluding singular values and computing it in future.
    #Tolerance is also,the percentage of pixels that can be a mismatch between two images.
    
    #Calculating the eigenvalues and the value of matrix V 
    eigs, V = linalg.eig(A.T.dot(A))

    #Singular values of a m*n matrix are the square root of the non-negative eigenvalues.
    sing_vals = np.sqrt(eigs)

    #Sorting both sigular values and matrix V in descending order so that the higher values are placed before the lower ones.
    idx = np.argsort(sing_vals)

    sing_vals = sing_vals[idx[::-1]]
    V = V[:, idx[::-1]]

    #Remove zero singular values below tolerance i.e values which would'nt benefit in determining the picture
    #The goal is to reduce the image and by removing the zero singular values, we would try to obtain an image with lowest ranks
    sing_vals_trunc = sing_vals[sing_vals>tol]
    V = V[:, sing_vals>tol]

    #It is not necessary to store the entire sigma matrix, so that only the diagonal is returned
    #So,sigma is calculated by extracting the zero singular values below tolerance
    sigma = sing_vals_trunc

    #Evaluating U matrix of dimension m*m using the numpy array of image and matrix V
    #Symbol @ is used for matrix multiplication
    U = A @ V /sing_vals_trunc
    
    #Returning the real values of matrix U, Sigma and the transpose of matrix V
    return U.real, sigma.real, V.T.real          


#Truncate is an user-defined function that helps us extract the useful first k columns from U matrix, the value of sigma and the first k rows from V matrix
#We do so because our goal is to compress the image and reduce the amount of rows and columns needed to construct the same image

def truncate(U, S, V, k):
      U_trunc = U[:, :k]
      S_trunc = S[:k]
      V_trunc = V[:k, :]
      return U_trunc, S_trunc, V_trunc


#Here, we assign the rank approximation to 50. 
k = 50

#Here, we assign the values of U, Sigma and transpose of V and apply user-defined function SVD on the numpy array of the image
U,S,V = svd(np.array(img))

#Now, extracting the useful values for U,S and V
U_trunc, S_trunc, V_trunc = truncate(U, S, V, k)

#Grayscale image is stored as 8 bits per pixel. 2^8 =256 so while reconstructing, it is multiplied by 256 to stay within range
reconstruct = 256 * U_trunc @ np.diag(S_trunc) @ V_trunc

plt.imshow(reconstruct,cmap='gray')
plt.show()


#Same thing repeated with rank approximation of 500
k = 500        
U,S,V = svd(np.array(img))
U_trunc, S_trunc, V_trunc = truncate(U, S, V, k)
reconstruct = 256 * U_trunc @ np.diag(S_trunc) @ V_trunc

plt.imshow(reconstruct,cmap='gray')
plt.show()


#Same thing repeated with rank approximation of 1000
k = 1000        
U,S,V = svd(np.array(img))
U_trunc, S_trunc, V_trunc = truncate(U, S, V, k)
reconstruct = 256 * U_trunc @ np.diag(S_trunc) @ V_trunc

plt.imshow(reconstruct,cmap='gray')
plt.show()
