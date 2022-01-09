from PIL import Image
import os, sys
# add path
from config import path_to_training_data
import tensorflow as tf

path = path_to_training_data
dirs = os.listdir( path )

def resize():
    for item in dirs:
        print (item)
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((70,70), Image.ANTIALIAS)
            x = tf.convert_to_tensor(im, dtype=tf.uint8)
            tf.image.resize_images(x, size=[14700, 14700])
            imResize.save(f+'.png', 'png', quality=70)

resize()

