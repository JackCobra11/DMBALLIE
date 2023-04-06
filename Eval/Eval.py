import numpy
import cv2
import vif,loe
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
i=573
folder="./result/"
im1=folder+str(i)+"_origin.jpg"
im2=folder+str(i)+"_n.jpg"
im3=folder+str(i)+"_dne.jpg"
im4=folder+str(i)+"_de.jpg"
img1=io.imread(im1,as_gray=True)
img2=io.imread(im2,as_gray=True)
img3=io.imread(im3,as_gray=True)
img4=io.imread(im4,as_gray=True)
im1=cv2.imread(im1)
im2=cv2.imread(im2)
im3=cv2.imread(im3)
im4=cv2.imread(im4)

psnr1=compare_psnr(im1,im2)
psnr2=compare_psnr(im1,im3)
psnr3=compare_psnr(im1,im4)

imssim1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
imssim2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
imssim3=cv2.cvtColor(im3,cv2.COLOR_BGR2GRAY)
imssim4=cv2.cvtColor(im4,cv2.COLOR_BGR2GRAY)
ssim1=compare_ssim(imssim1,imssim2)
ssim2=compare_ssim(imssim1,imssim3)
ssim3=compare_ssim(imssim1,imssim4)

imab1=cv2.cvtColor(im1,cv2.COLOR_BGR2HSV)
imab2=cv2.cvtColor(im2,cv2.COLOR_BGR2HSV)
imab3=cv2.cvtColor(im3,cv2.COLOR_BGR2HSV)
imab4=cv2.cvtColor(im4,cv2.COLOR_BGR2HSV)
imab1=imab1[:,:,2].ravel()
imab2=imab2[:,:,2].ravel()
imab3=imab3[:,:,2].ravel()
imab4=imab4[:,:,2].ravel()
ab1=numpy.mean(imab1)-numpy.mean(imab2)
ab2=numpy.mean(imab1)-numpy.mean(imab3)
ab3=numpy.mean(imab1)-numpy.mean(imab4)


print("Origin-to-Dark[PSNR]:",psnr1,";Origin-to-Enhanced[PSNR]:",psnr2)
print("Origin-to-Enhanced2[PSNR]:",psnr3)
#print("Increase ratio PSNR:",(psnr2/psnr1))
print("Origin-to-Dark[SSIM]:",ssim1,";Origin-to-Enhanced[PSNR]:",ssim2)
print("Origin-to-Enhanced2[PSNR]:",ssim3)
#print("Increase ratio SSIM:",(ssim2/ssim1))
print("Dark[AB]:",ab1,";Enhanced[AB]:",ab2)
print("Enhanced2[AB]:",ab3)
#print("Decrease ratio abs(AB):",(numpy.abs(ab1)/numpy.abs(ab2)))

vif1 = vif.compare_vif(img1, img2)
vif2 = vif.compare_vif(img1, img3)
vif3 = vif.compare_vif(img1, img4)
print("Origin-to-Dark[VIF]:",vif1,";Origin-to-Enhanced[VIF]:",vif2)
print("Origin-to-Enhanced2[VIF]:",vif3)
#print("Increase ratio VIF:",(vif2/vif1))
loe1= loe.loe(im1,im2)
loe2= loe.loe(im1,im3)
loe3= loe.loe(im1,im4)
print("Origin-to-Dark[LOE]:",loe1,";Origin-to-Enhanced[LOE]:",loe2)
print("Origin-to-Enhanced2[LOE]:",loe3)
#print("Decrease ratio LOE:",(loe1/loe2))

