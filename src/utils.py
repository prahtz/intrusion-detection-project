import cv2
import numpy as np

def l1_dist(im1, im2):
    res = np.abs(im1-im2)
    return res if len(im1.shape) < 3 else np.sum(res, axis = -1)
def l2_dist(im1, im2):
    res = (im1-im2)**2
    return np.sqrt(res) if len(im1.shape) < 3 else np.sqrt(np.sum(res, axis=-1))
def linf_dist(im1, im2):
    res = np.abs(im1-im2)
    return res if len(im1.shape) < 3 else np.max(res, axis = -1)


def cos_similarity(v1, v2):
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def gradient_magnitudes(mask, frame, pts):
    sobelx = cv2.Sobel(mask,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(mask,cv2.CV_64F,0,1,ksize=5)
    gd_mask = np.sqrt((sobelx ** 2) + (sobely ** 2))
    gd_mask = gd_mask[[p[0] for p in pts], [p[1] for p in pts]]

    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
    gd_frame = np.sqrt((sobelx ** 2) + (sobely ** 2))
    gd_frame = gd_frame[[p[0] for p in pts], [p[1] for p in pts]]
    return cos_similarity(gd_mask, gd_frame)

def steger_similarity(v1, v2):
    a = 0.0
    for i in range(0, v1.shape[0]):
        a += np.abs(v1[i].dot(v2[i]))
    return a/v1.shape[0]

def gradient_directions(mask, frame, pts):
    sobelx = cv2.Sobel(mask,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(mask,cv2.CV_64F,0,1,ksize=5)
    mag_mask = np.sqrt((sobelx ** 2) + (sobely ** 2))
    gd_mask = np.stack([np.divide(sobelx, mag_mask, out=np.zeros_like(sobelx), where=mag_mask!=0), np.divide(sobely, mag_mask, out=np.zeros_like(sobely),where=mag_mask!=0)], axis = -1)
    gd_mask = gd_mask[[p[0] for p in pts], [p[1] for p in pts]]

    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
    mag_frame = np.sqrt((sobelx ** 2) + (sobely ** 2))
    gd_frame = np.stack([np.divide(sobelx, mag_frame, out=np.zeros_like(sobelx), where=mag_frame!=0), np.divide(sobely, mag_frame, out=np.zeros_like(sobely), where=mag_frame!=0)], axis = -1)
    gd_frame = gd_frame[[p[0] for p in pts], [p[1] for p in pts]]
    return steger_similarity(gd_mask, gd_frame)


