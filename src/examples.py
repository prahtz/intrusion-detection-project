import cv2
import numpy as np
from intrusion_detection import start_analysis


def example1():
    video_name = 'res/video.avi'
    text_out, graphical = start_analysis(video_name, intensity_threshold = 25, alpha = 0.1, beta = 0.4, update_fp = False, visualize = False)

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

example1()