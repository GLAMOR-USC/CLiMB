import os
import os.path as op
from PIL import Image
import numpy as np
from tqdm import tqdm

def resize_image(image, desired_shape):
    try:
        w, h = image.size
        #logging.info("size: {}".format((w, h)))
        if image.mode == 'CMYK':
            image = image.convert('RGB')

        if w > h:
            d_w = max(desired_shape)
            d_h = min(desired_shape)
            #print("d_w: {} d_h: {}".format(d_w, d_h))
            if w >= d_w:
                new_h = int(h * d_w / w)
                #print("new_h: {}".format(new_h))
                if new_h > d_h:
                    image = image.resize((int(w * d_h/h), d_h), resample=0)
                else:
                    image = image.resize((d_w, new_h), resample=0)
            else:
                if h > d_h:
                    new_w = int(d_h * w / h)
                    #print("new_w: {}".format(new_w))
                    image = image.resize((new_w, d_h), resample=0)
        else:
            d_h = max(desired_shape)
            d_w = min(desired_shape) 
            #print("d_w: {} d_h: {}".format(d_w, d_h))
            if h >= d_h:
                new_w = int(w * d_h / h)
                #print("new_w: {}".format(new_w))
                if new_w > d_w:
                    image = image.resize((d_w, int(h * d_w/w)), resample=0)
                else:
                    image = image.resize((new_w, d_h), resample=0)
            else:
                if w > d_w:
                    new_h = int(d_w * h / w)
                    #print("new_h: {}".format(new_h))
                    image = image.resize((d_w, new_h), resample=0)

        image_arr = np.asarray(image)               # size: (w, h) -> shape (h, w)
        if len(image_arr.shape) < 3:                # for grayscale images, stack channels
            image_arr = np.stack((image_arr,)*3, axis=-1)
        elif len(image_arr.shape) == 3 and image_arr.shape[2] > 3:
            image_arr = image_arr[:, :, :3]
        padded_image = np.zeros((d_h, d_w, 3,), dtype=np.float64)
        padded_image[:image_arr.shape[0], :image_arr.shape[1]] = image_arr
        return padded_image
    except Exception as e:
        d_w = max(desired_shape)
        d_h = min(desired_shape)
        padded_image = np.zeros((d_h, d_w, 3,), dtype=np.float64)
        return padded_image