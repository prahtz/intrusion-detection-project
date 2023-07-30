import cv2
import time
from background_subtraction import BackgroundSubtraction

video_name = "data/video.avi"
cap = cv2.VideoCapture(video_name)

background_samples = []
num_background_samples = 20
i = 0
while cap.isOpened() and i < num_background_samples:
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        print("Released Video Resource")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background_samples.append(frame)
    i += 1


def area_filter_fn(area):
    return area > 100


bkg_sub = BackgroundSubtraction(
    background_samples=background_samples,
    threshold=15.0,
    alpha=0.1,
    beta=0.1,
    opening_k_shape=(7, 7),
    closing_k_shape=(13, 13),
    handle_light_changes=True,
    zncc_threshold=0.95,
    area_filter_fn=area_filter_fn,
)

while cap.isOpened():
    ret, color_frame = cap.read()
    if not ret or frame is None:
        cap.release()
        print("Released Video Resource")
        break
    start = time.time()
    frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    bboxes = bkg_sub.step(frame)
    end = time.time()
    print("Subtraction time:", end - start)

    for bbox in bboxes:
        x, y, w, h = bbox
        color_frame = cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", color_frame)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
