import numpy
import cv2
import vif,loe
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
from glob import glob
import cv2
import argparse

def eval(arg):
    origin_folder = arg.origin
    path = glob(origin_folder+'/*.*')
    input_folder = arg.origin
    path2 = glob(input_folder+'/*.*')
    psnr_result = []
    ssim_result = []
    ab_result = []
    vif_result = []
    loe_result = []
    for i in range(len(path)):
        im1_path = path[i]
        im2_path = path2[i]
        img1=io.imread(im1_path,as_gray=True)
        img2=io.imread(im2_path,as_gray=True)
        
        im1=cv2.imread(im1_path)
        im2=cv2.imread(im2_path)
        psnr1=compare_psnr(im1,im2)
        psnr_result.append(psnr1)

        imssim1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        imssim2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
        ssim1=compare_ssim(imssim1,imssim2)
        ssim_result.append(ssim1)

        imab1=cv2.cvtColor(im1,cv2.COLOR_BGR2HSV)
        imab2=cv2.cvtColor(im2,cv2.COLOR_BGR2HSV)
        imab1=imab1[:,:,2].ravel()
        imab2=imab2[:,:,2].ravel()
        ab1=numpy.mean(imab1)-numpy.mean(imab2)
        ab_result.append(ab1)

        vif1 = vif.compare_vif(img1, img2)
        vif_result.append(vif1)

        loe1= loe.loe(im1,im2)
        loe_result.append(loe1)
    psnr = numpy.mean(psnr_result)
    ssim = numpy.mean(ssim_result)
    ab = numpy.mean(ab_result)
    vif = numpy.mean(vif_result)
    loe = numpy.mean(loe_result)
    return(psnr,ssim,ab,vif,loe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", "-o", type=str, default='./test', help='test image folder')
    parser.add_argument("--input", "-i", type=str, default='./result', help='result folder')
    arg = parser.parse_args()
    psnr,ssim,ab,vif,loe = eval(arg)
    print('psnr:',psnr)
    print('ssim:',ssim)
    print('ab:',ab)
    print('vif:',vif)
    print('loe:',loe)
    
