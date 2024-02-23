import cv2
import numpy as np
from matplotlib.image import imread 
import matplotlib.pyplot as plt 

path='C:\ProjectMAI\doraemon.jpg'

def convertImageToMatrix(imagePath):
    a=cv2.imread(imagePath,0)
    return a

def showImage(imagePath):
    a=cv2.imread(imagePath,1)
    cv2.imshow('doraemon',a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def convertMatrixToImage(matrix):
    cv2.imwrite('SVD.jpg',matrix)
    cv2.imshow("SVD", matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def AcolumMatrix(matrix,index):
    a=[]
    for i in range(0,len(matrix)):
            a.append(matrix[i][index-1])
    return np.matrix(a)
def ALinematrix(matrix,index):
    return np.matrix(matrix[index-1])

def approximationK(Matrix,k):
    U,S,VT=np.linalg.svd(Matrix,full_matrices=True)
    S=np.diag(S)
    print(S)
    ApproximationKMatrix=0
    for i in range(1,k+1):
        ApproximationKMatrix+=np.dot(np.dot(S[i-1,i-1],AcolumMatrix(U,i).T),ALinematrix(VT,i))
    return ApproximationKMatrix

convertMatrixToImage(approximationK(convertImageToMatrix(path),5))
# showImage(path)



