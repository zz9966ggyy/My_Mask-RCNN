import cv2   
imgFile = "D:\\bishe\\aaaa\\MY_Mask_RCNN-master\images\\12283150_12d37e6389_z.jpg"# 读取文件的路径
img3 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR
saveFile = "D:\\bishe\\aaaa\MY_Mask_RCNN-master\\videos\\save2\\1.jpg"  # 保存文件的路径
 # cv2.imwrite(saveFile, img3, [int(cv2.IMWRITE_PNG_COMPRESSION), 8])  # 保存图像文件, 设置压缩比为 8
cv2.imwrite(saveFile, img3)
