from glob import glob
import matplotlib.pyplot as plt



data_path = glob('./data/training/image/*')
label_path = glob('./data/training/semantic_rgb/*')


plt.figure(figsize=(16,9))     
for i in range(3):
    plt.subplot(2,3,i+1)  
    img = plt.imread(data_path[i])
    plt.imshow(img)
for i in range(3):
    plt.subplot(2,3,i+4)  
    img = plt.imread(label_path[i])
    plt.imshow(img)
plt.show()