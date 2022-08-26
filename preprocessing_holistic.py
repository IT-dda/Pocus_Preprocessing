import cv2
import os
from tqdm import tqdm
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BG_COLOR = (0, 0, 0)  # black

BASE_DIR = "./upper_pose/"
ORIGINAL_DIR = os.path.join(BASE_DIR, "original")
POSE_DIR = os.path.join(BASE_DIR, "holistic")
if not os.path.isdir(POSE_DIR):
    os.mkdir(POSE_DIR)
CLASSES = ["correct", "turtle", "shoulder-left", "shoulder-right", "head-left", "head-right", "chin-left", "chin-right"]

for i in range(8):
    print(f"index {i} is running...")

    # 이미지 파일명 리스트 가져오기
    i_dir = os.path.join(ORIGINAL_DIR, str(i))  # ./upper_pose/original/N
    img_list = [img for img in os.listdir(i_dir) if img.endswith("jpg")]

    # 인덱스별 폴더 생성 : ./upper_pose/pose/N
    i_dir_pose = os.path.join(POSE_DIR, CLASSES[i])
    if not os.path.isdir(i_dir_pose):
        os.mkdir(i_dir_pose)

    with mp_holistic.Holistic(
        static_image_mode=True, model_complexity=2, enable_segmentation=True, refine_face_landmarks=True
    ) as holistic:
        for file in tqdm(img_list):
            read_path = os.path.join(i_dir, file)
            write_path = os.path.join(i_dir_pose, file)

            # bgr > gray scale
            img_bgr = cv2.imread(read_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # padding
            h, w = img_gray.shape
            dif = w - h
            img = cv2.copyMakeBorder(img_gray, dif // 2, dif // 2, 0, 0, cv2.BORDER_CONSTANT, value=BG_COLOR)

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            results = holistic.process(image)

            # Draw face detections of each face.
            if not results.pose_landmarks:
                continue

            annotated_image = image.copy()

            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # bg_image = np.zeros(image.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            # annotated_image = np.where(condition, bg_image, bg_image)
            annotated_image[:] = BG_COLOR

            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            cv2.imwrite(write_path, annotated_image)
