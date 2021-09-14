import cv2
import numpy as np
from intrusion_detection import start_analysis


def example1():
    video_name = 'res/video.avi'
    text_out, graphical = start_analysis(video_name, intensity_threshold = 25, alpha = 0.1, beta = 0.4, closing_k_shape = (23,23), update_fp = False, visualize = False)

    for out, bkg, frame, text in zip(graphical['output'], graphical['background'], graphical['frame'], text_out):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        cv2.namedWindow("bkg", cv2.WINDOW_NORMAL)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow('output', out)
        cv2.imshow('bkg', bkg.astype(np.uint8))
        cv2.imshow('frame', frame)
        print(text)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

def testZNCC():
    video_name = 'res/video.avi'
    start_analysis(video_name, intensity_threshold = 15, alpha = 0.3, beta = 0.4, closing_k_shape = (11,11), update_fp = False, visualize = True)

testZNCC()