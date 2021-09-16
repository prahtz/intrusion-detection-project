import cv2
import numpy as np
from intrusion_detection import start_analysis
import prettytable


def analyze_and_vizualize():
    video_name = 'res/video.avi'
    text_out, graphical = start_analysis(video_name, intensity_threshold = 15, alpha = 0.3, beta = 0.4, closing_k_shape = (13,13), update_fp = False, visualize = False)
    f = open("output.txt", "w")
    for out, text in zip(graphical, text_out):
        result = prettytable.PrettyTable(["Frame number", "Detected objects"])
        result.add_row([str(text['frame_number']), str(text['obj_number'])])
        result = result.get_string() + '\n'
        obj_table = prettytable.PrettyTable(["Object ID", "Color", "Area", "Classification"])
        for obj in text['objects']:
            obj_table.add_row([obj['id'], obj['color'], obj['area'], obj['classification']])
        result += obj_table.get_string() + '\n\n'
        f = open("output.txt", "a")
        f.write(result)
        f.close()

        cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        cv2.namedWindow("bkg", cv2.WINDOW_NORMAL)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("bkg_subtraction", cv2.WINDOW_NORMAL)
        cv2.namedWindow("noise_reduction", cv2.WINDOW_NORMAL)  
        cv2.namedWindow("morph", cv2.WINDOW_NORMAL)
        cv2.namedWindow("light_changes", cv2.WINDOW_NORMAL) 
        cv2.imshow('output', out['output'])
        cv2.imshow('bkg', out['background'])
        cv2.imshow('frame', out['frame'])
        cv2.imshow('bkg_subtraction', out['subtraction'])
        cv2.imshow('noise_reduction', out['noise'])
        cv2.imshow('morph', out['morph'])
        cv2.imshow('light_changes', out['light'])
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

def visualize_runtime():
    video_name = 'res/video.avi'
    start_analysis(video_name, intensity_threshold = 15, alpha = 0.3, beta = 0.4, closing_k_shape = (13,13), update_fp = False, visualize = True)

def analyze_and_store():
    video_name = 'res/video.avi'
    text_out, _ = start_analysis(video_name, intensity_threshold = 15, alpha = 0.3, beta = 0.4, closing_k_shape = (13,13), update_fp = False, visualize = False)
    f = open("output.txt", "w")
    for text in text_out:
        result = prettytable.PrettyTable(["Frame number", "Detected objects"])
        result.add_row([str(text['frame_number']), str(text['obj_number'])])
        result = result.get_string() + '\n'
        obj_table = prettytable.PrettyTable(["Object ID", "Color", "Area", "Classification"])
        for obj in text['objects']:
            obj_table.add_row([obj['id'], obj['color'], obj['area'], obj['classification']])
        result += obj_table.get_string() + '\n\n'
        f = open("output.txt", "a")
        f.write(result)
        f.close()

analyze_and_vizualize()