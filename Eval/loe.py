import cv2
import numpy as np

def loe(I, Ie):
    N, M, n = I.shape
    L = np.max(I, axis=2)
    Le = np.max(Ie, axis=2)
    r = 50 / min(M, N)
    Md = round(M*r)
    Nd = round(N*r)
    Ld = cv2.resize(L, (Md, Nd))
    Led = cv2.resize(Le, (Md, Nd))
    RD = np.zeros((Nd, Md))
    for y in range(Md):
        for x in range(Nd):
            E = np.logical_xor((Ld[x, y] >= Ld), (Led[x, y] >= Led))
            RD[x, y] = np.sum(E)
    LOE = np.sum(RD) / (Md*Nd)
    return LOE

# if __name__ == '__main__':
#     I = cv2.imread('84_origin.jpg')  # original image
#     Ie = cv2.imread('84_enhanced.jpg')  # enhanced image
#     LOE = loe(I, Ie)
#     print('LOE:', LOE)
