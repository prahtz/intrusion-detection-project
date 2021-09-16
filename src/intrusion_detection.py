import cv2
import numpy as np
from utils import *
from background_operations import *

COLORS_PATTERN_DEFAULT = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow': (0, 255, 255), 'cyan':(255,255,0), 'magenta':(255,0,255)}

def get_binary_image(bkg, frame, distance, threshold):
    binary = distance(bkg, frame)
    binary[binary <= threshold] = 0
    binary[binary > threshold] = 255
    return binary

def remove_noise(binary, noise_threshold):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.int8))
    for i in range(1,n):
        if stats[i, -1] < noise_threshold:
            labels[labels == i] = 0
    result = binary.copy()
    result[labels == 0] = 0
    result[labels != 0] == 255
    return result

def apply_morphological_operations(binary, closing_k_shape):
    morphed = binary.copy()
    closing_k = cv2.getStructuringElement(cv2.MORPH_RECT, closing_k_shape)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, closing_k)
    morphed = morphed.astype(np.uint8)
    
    contour, _ = cv2.findContours(morphed,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cnt = cv2.approxPolyDP(cnt, 3, True)
        cv2.drawContours(morphed,[cnt],0,255,-1)
    return morphed

def get_connected_components(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.int8))
    return n-1, labels, stats[1:]

def remove_light_changes(change_mask, bkg, frame, thresh = 0.95):
    result = change_mask.copy()
    contours, _ = cv2.findContours(change_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        bkg_patch = bkg[y:y+h, x:x+w].copy()
        frame_patch = frame[y:y+h, x:x+w].copy()
        bkg_patch[change_mask[y:y+h, x:x+w] == 0] = 0
        frame_patch[change_mask[y:y+h, x:x+w] == 0] = 0
        zncc = cv2.matchTemplate(bkg_patch, frame_patch, cv2.TM_CCOEFF_NORMED)
        if zncc > thresh:
            result[y:y+h, x:x+w] = 0
    return result
        

def classify_objects(frame, labels, n, stats, person_thresh = 3000, obj_thresh = 0.8):
    obj = { i: 'other' for i in range(0, n)}
    max_area_id = np.argmax(stats[:,-1]) if n > 0 else -1
    if max_area_id != -1 and stats[max_area_id, -1] > person_thresh:
        obj[0] = 'person'
    for i in range(0, n):
        if obj[i] != 'person':
            region = np.zeros(labels.shape, dtype=np.uint8)
            region[labels == i+1] = 255
            edge_pts = np.argwhere(cv2.Canny(region, 100, 200))
            if gradient_directions(region, frame, edge_pts) < obj_thresh:
                obj[i] = 'false object'
            else:
                obj[i] = 'true object'
    return obj

def colored_blobs(binary, n, labels, colors_pattern = COLORS_PATTERN_DEFAULT):
    blobs = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if n > len(colors_pattern):
        for i in range(len(colors_pattern), n):
            colors_pattern['random_' + str(i)] = tuple(np.random.randint(50,200,(1,3)))
    colors = [key for key, _ in colors_pattern.items()]
    for i in range(0, n):
        blobs[labels == i+1] = colors_pattern[colors[i]]
    return blobs, colors

def start_analysis(capture_id, train_frames = 100, intensity_threshold = 25, intensity_measure = linf_dist, alpha = 0.1, beta = 0.4, closing_k_shape = (13,13), update_fp = False, visualize = False):
    cap = cv2.VideoCapture(capture_id)
    bkg, _ = blind_background(cap, train_frames)
    text_results = []
    graphical_results = []
    frame_number = 0
    noise_threshold = 100
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            print("Released Video Resource")
            break
        frame_number += 1
        frame = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        #Background subtraction
        binary = get_binary_image(bkg, frame, intensity_measure, intensity_threshold)

        #Remove small area blobs
        binary_noise = remove_noise(binary, noise_threshold)

        #Initialize change mask by applying some binary morphological operators to the binary image
        binary_morph = apply_morphological_operations(binary_noise, closing_k_shape)

        #Removes from the change mask the blobs having a high intensity affinity with the background
        change_mask = remove_light_changes(binary_morph, bkg, frame)

        #Find connected components (blobs) with statistics
        n, labels, stats = get_connected_components(change_mask)

        #Background update
        bkg = update_background(change_mask, bkg, frame, alpha)

        #Classify objects
        obj = classify_objects(frame, labels, n, stats)

        #Compute colored blobs
        blobs, colors = colored_blobs(change_mask, n, labels)

        #Update text results
        text_results.append({
                            'frame_number':frame_number, 
                            'obj_number': n,
                            'objects': [{'id': i+1, 'color': colors[i], 'area':stats[i,-1], 'classification': obj[i]} for i in range(0, n)]
                            })
        #Update background using false positive objects
        if update_fp:
            bkg = update_background_fp(obj, n, labels, bkg, frame, beta)
        
        if visualize:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL) 
            cv2.namedWindow("bkg", cv2.WINDOW_NORMAL)
            cv2.namedWindow("real_frame", cv2.WINDOW_NORMAL) 
            cv2.namedWindow("bkg_subtraction", cv2.WINDOW_NORMAL)
            cv2.namedWindow("noise_reduction", cv2.WINDOW_NORMAL)  
            cv2.namedWindow("morph", cv2.WINDOW_NORMAL)
            cv2.namedWindow("light_changes", cv2.WINDOW_NORMAL) 
            cv2.imshow('result', blobs)
            cv2.imshow('bkg', bkg.astype(np.uint8))
            cv2.imshow('real_frame', frame.astype(np.uint8))
            cv2.imshow('bkg_subtraction', binary)
            cv2.imshow('noise_reduction', binary_noise)
            cv2.imshow('morph', binary_morph)
            cv2.imshow('light_changes', change_mask)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        graphical_results.append({'background': bkg.copy().astype(np.uint8),
                                    'frame' : frame.copy().astype(np.uint8),
                                    'output' : blobs.copy().astype(np.uint8),
                                    'subtraction' : binary.copy(),
                                    'noise' : binary_noise.copy(),
                                    'morph': binary_morph.copy(),
                                    'light': change_mask.copy()
                                })
    return text_results, graphical_results

