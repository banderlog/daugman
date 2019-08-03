import cv2
import numpy as np
import itertools


def daugman(center: Tuple[int, int], start_r: int,
            gray_img: 'np.array') -> 'Tuple[float, List[Tuple[int, int], int]]':
    """return maximal intense radius for given center
    center -- tuple(x, y)
    start_r -- int
    gray_img -- grayscale picture as np.array(), it should be square
    
    output: intensity_value, center_coords, radius
    """
    # get separate coordinates
    x, y = center
    # get img dimensions
    h, w = gray_img.shape
    # for calculation convinience
    img_shape = np.array([h, w])
    c = np.array(center)
    # define some other vars
    tmp = []
    mask = np.zeros_like(gray_img)

    # for every radius in range
    # we are presuming that iris will be no bigger than 1/3 of picture
    for r in range(start_r, int(h/3)):
        # draw circle on mask
        cv2.circle(mask, center, r, 255, 1)
        # get pixel from original image
        radii = gray_img & mask  # it is faster than np or cv2
        # normalize
        tmp.append(radii[radii > 0].sum()/(2*3.1415*r))
        # refresh mask
        mask.fill(0)
        
    # calculate delta of radius intensitiveness
    tmp = np.array(tmp)
    tmp = tmp[1:] - tmp[:-1]
    # aply gaussian filter
    tmp = abs(cv2.GaussianBlur(tmp[:-1], (1, 5), 0))
    # get maximum value
    idx = np.argmax(tmp)
    # return value, center coords, radius
    return tmp[idx], [center, idx + start_r]
    

def find_iris(gray: 'np.array',
              start_r: int) -> 'np.array(Tuple[int, int], int)':
    """Apply daugman() on every pixel in calculated image slice
        gray -- graysacale img as np.array()
        start_r -- initial radius as int
    Selection of image slice guarantees that every
    radius will be drawn in iage borders, so we need to check it (speed up)
    
    output: radius with biggest intensiveness delta on image
            as [(xc, yc), radius]
    
    To speed up the whole process we need to pregenerate all centers for detection
    """
    _, s = gray.shape
    # reduce step for better accuracy
    # 's/3' is the maximum radius of a daugman() search
    a = range(0 + int(s/3), s - int(s/3), 3)
    all_points = list(itertools.product(a, a))
    
    values = []
    coords = []
    
    for p in all_points:
        tmp = daugman(p, start_r, gray)
        if tmp is not None:
            val, circle = tmp
            values.append(val)
            coords.append(circle)
    
    # return the radius with biggest intensiveness delta on image
    # [(xc, yc), radius]
    return coords[np.argmax(values)]
