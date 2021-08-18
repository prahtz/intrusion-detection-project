import cv2
import numpy as np
from background_operations import *
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

def l1_dist(im1, im2):
    res = np.abs(im1-im2)
    return res if len(im1.shape) < 3 else np.sum(res, axis = -1)
def l2_dist(im1, im2):
    res = (im1-im2)**2
    return np.sqrt(res) if len(im1.shape) < 3 else np.sqrt(np.sum(res, axis=-1))
def linf_dist(im1, im2):
    res = np.abs(im1-im2)
    return res if len(im1.shape) < 3 else np.max(res, axis = -1)

VIDEO_NAME = 'res/video.avi'


cap = cv2.VideoCapture(VIDEO_NAME)
bkg, samples = blind_background(cap)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
text_results = []

area_thresh = 3000
obj_thresh = 0.75
T = 15
alpha = 0.3
distance = linf_dist
frame_number = 1
colors_pattern = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow': (0, 255, 255), 'cyan':(255,255,0), 'magenta':(255,0,255)}
while(cap.isOpened()):
    ret, frame = cap.read()
    if(not ret or frame is None):
        cap.release()
        print("Released Video Resource")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype(float)
    
    #print(frame.shape)
    dist = distance(bkg, frame)
    new_f = dist.copy().astype(np.uint8)
    new_f[new_f <= T] = 0
    new_f[new_f > T] = 255
    #Remove noise using opening
    n, labels, stats, _ = cv2.connectedComponentsWithStats(new_f.astype(np.int8))
    A = 100
    for i in range(1,n):
        if stats[i, -1] < A:
            labels[labels == i] = 0
    new_f[labels == 0] = 0
    new_f[labels != 0] == 255

    #Initialize change mask and closing operation to remove false negatives and avoid undesired background updates
    change_mask = new_f.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(55,55))
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
    #change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

    #Background update
    f_mask = change_mask == 255
    b_mask = change_mask == 0
    x = np.zeros(bkg.shape)
    y = np.zeros(bkg.shape)
    x[b_mask] = frame[b_mask]
    y[b_mask] = bkg[b_mask]
    app = x*alpha + (1-alpha)*y
    app[f_mask] = bkg[f_mask]
    bkg = app.copy()
    #Enhance detection frame


    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(23,23))
    #new_f = cv2.morphologyEx(new_f, cv2.MORPH_CLOSE, kernel)
    new_f = new_f.astype(np.uint8)


    n, labels, stats, _ = cv2.connectedComponentsWithStats(change_mask.astype(np.int8))
    stats = stats[1:]
    n -= 1
    obj = { i: 'other' for i in range(0, n)}
    max_area_id = np.argmax(stats[:,-1]) if n > 0 else -1
    if max_area_id != -1 and stats[max_area_id, -1] > area_thresh:
        obj[0] = 'person'
    for i in range(0, n):
        if obj[i] != 'person':
            region = np.zeros(labels.shape, dtype=np.uint8)
            region[labels == i+1] = 255
            #contours = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            edge_pts = np.argwhere(cv2.Canny(region, 100, 200))
            if gradient_magnitudes(region, frame,  edge_pts) < obj_thresh:
                obj[i] = 'false object'
            else:
                obj[i] = 'true object'
    new_f = cv2.cvtColor(change_mask, cv2.COLOR_GRAY2BGR)

    if n > len(colors_pattern):
        for i in range(len(colors_pattern), n):
            colors_pattern['random_' + str(i)] = tuple(np.random.randint(50,200,(1,3)))
    colors = [key for key, _ in colors_pattern.items()]
    for i in range(0, n):
        new_f[labels == i+1] = colors_pattern[colors[i]]
    text_results.append({
                        'frame_number':frame_number, 
                        'obj_number': n,
                        'objects': [{'id': i+1, 'color': colors[i], 'area':stats[i,-1], 'classification': obj[i]} for i in range(0, n)]
                        })

    '''for i in range(0, n):
        if obj[i] == 'false object':
            o_mask = labels == i+1
            x = np.zeros(bkg.shape)
            y = np.zeros(bkg.shape)
            x[o_mask] = frame[o_mask]
            y[o_mask] = bkg[o_mask]
            app = x*alpha + (1-alpha)*y
            bkg[o_mask] = app[o_mask]'''

    print(text_results[frame_number-1])

    frame_number+=1

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("bkg", cv2.WINDOW_NORMAL)
    cv2.namedWindow("change_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("real_frame", cv2.WINDOW_NORMAL) 
    cv2.imshow('frame', new_f)
    cv2.imshow('change_mask', change_mask.astype(np.uint8))
    cv2.imshow('bkg', bkg.astype(np.uint8))
    cv2.imshow('real_frame', frame.astype(np.uint8))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break