import cv2
import numpy as np

def blind_background(cap, frame_limit = 100, interpolation = np.median):
    bkg = []
    for i in range(0, frame_limit):
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Frame limit is to high')
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bkg.append(frame)
        
    samples = np.stack(bkg, axis = 0)
    return np.float32(interpolation(samples, axis = 0)), samples

def update_background(change_mask, bkg, frame, alpha):
    f_mask = change_mask == 255
    b_mask = change_mask == 0
    x = np.zeros(frame.shape)
    y = np.zeros(bkg.shape)
    x[b_mask] = frame[b_mask]
    y[b_mask] = bkg[b_mask]
    new_bkg = x*alpha + (1-alpha)*y
    new_bkg[f_mask] = bkg[f_mask]
    return np.float32(new_bkg)

def update_background_fp(obj, n, labels, bkg, frame, beta):
    for i in range(0, n):
        if obj[i] == 'false object':
            o_mask = labels == i+1
            x = np.zeros(bkg.shape)
            y = np.zeros(bkg.shape)
            x[o_mask] = frame[o_mask]
            y[o_mask] = bkg[o_mask]
            app = x*beta + (1-beta)*y
            bkg[o_mask] = app[o_mask]
    return bkg

