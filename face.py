import cv2
import os
from tqdm import tqdm
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


BASE_DIR = "./upper_pose/"
GRAY_DIR = os.path.join(BASE_DIR, "gray")
FACE_DIR = os.path.join(BASE_DIR, "face")
if not os.path.isdir(FACE_DIR):
    os.mkdir(FACE_DIR)

for i in range(8):
    print(f"index {i} is running...")

    # 이미지 파일명 리스트 가져오기
    i_dir = os.path.join(GRAY_DIR, str(i))  # ./upper_pose/gray/N
    img_list = [img for img in os.listdir(i_dir) if img.endswith("jpg")]

    # 인덱스별 폴더 생성 : ./upper_pose/face/N
    i_dir_face = os.path.join(FACE_DIR, str(i))
    if not os.path.isdir(i_dir_face):
        os.mkdir(i_dir_face)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        for file in tqdm(img_list):
            read_path = os.path.join(i_dir, file)
            save_path = os.path.join(i_dir_face, file)

            image = cv2.imread(read_path)

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            # cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # gray scale이라 필요없을거 같음
            results = face_detection.process(image)

            # Draw face detections of each face.
            if not results.detections:
                continue

            annotated_image = image.copy()

            for detection in results.detections:
                mp_drawing.draw_detection(annotated_image, detection)

            cv2.imwrite(save_path, annotated_image)
