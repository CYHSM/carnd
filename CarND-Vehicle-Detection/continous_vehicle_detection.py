import numpy as np
from moviepy.editor import VideoFileClip
import pickle
import vehicle_detection as vd
import cv2
from scipy.ndimage.measurements import label


def set_globals():
    global svc, X_scaler, boxes, smoothing_counter
    svc = pickle.load(open("svc.p", "rb"))
    X_scaler = pickle.load(open("X_scaler.p", "rb"))
    boxes = []
    smoothing_counter = 0


def process_video(video_path, file_out):
    """
    Finds lane lines for given video

    Inputs:
    - video_path : Path to video. o really.-
    """
    set_globals()
    output = file_out
    clip1 = VideoFileClip(video_path)
    # NOTE: this function expects color images!!
    clip = clip1.fl_image(process_frame)
    clip.write_videofile(output, audio=False)


def process_frame(img):
    """
    Processes one frame of an image

    Inputs:
    - img : Frame as color image
    """
    global smoothing_counter

    smoothing_window = 20
    # First find cars in the image
    draw_img, bbox_list = vd.find_cars(img, svc=svc, X_scaler=X_scaler, orient=9, pix_per_cell=8, cell_per_block=2,
                                       spatial_size=(16, 16), hist_bins=16, scale=1.5)
    # Add heat map
    boxes.append(bbox_list)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    if smoothing_counter > smoothing_window:
        for i in range(0, smoothing_window):
            heat = vd.add_heat(heat, boxes[-i])
        heat = vd.apply_threshold(heat, 1*smoothing_window)
    else:
        heat = vd.add_heat(heat, bbox_list)
        heat = vd.apply_threshold(heat, 1)
    smoothing_counter += 1

    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = vd.draw_labeled_bboxes(np.copy(img), labels)

    return draw_img

