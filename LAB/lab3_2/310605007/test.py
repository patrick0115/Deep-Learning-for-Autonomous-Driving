from PIL import Image
import cv2
from yolo import YOLO
# from IPython.display import Image as Ip
import os
yolo = YOLO()
img_path = "./img"

write_path = "./result"
labels_path = "./test_labels"
f = open("test_img.txt")
for line in f.readlines():
    img = os.path.join(img_path,line.strip()+".jpg")
    print(img)
    new = os.path.join(write_path,line.strip()+".jpg")
    try:
        image1 = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        r_image, labels = yolo.detect_image(image1)
        f2 = open(os.path.join(labels_path,line.strip()+".txt"),"w")
#         print(len(labels))
        for ll in labels:
#             print(ll)
            f2.writelines(ll+'\n')
        f2.close()
        r_image.save(new)
f.close()
