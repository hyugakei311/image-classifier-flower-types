import numpy as np
from PIL import Image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    w, h = image.size
    if w < h:
        ratio = 256 / w
        size = (256, h*ratio)
    elif w > h:
        ratio = 256/ h
        size = (w*ratio, 256)
    else:
        size = (256,256)
        
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((image.size[0]/2-112, image.size[1]/2-112, image.size[0]/2+112, image.size[1]/2+112))
    np_image = np.array(image)
    np_image = np_image/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2,0,1))
    return np_image