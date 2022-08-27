import cv2
import os
from tqdm import tqdm
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BG_COLOR = (0, 0, 0)  # black

BASE_DIR = "./upper_pose/"
ORIGINAL_DIR = os.path.join(BASE_DIR, "original")
POSE_DIR = os.path.join(BASE_DIR, "hands")
if not os.path.isdir(POSE_DIR):
    os.mkdir(POSE_DIR)
CLASSES = ["correct", "turtle", "shoulder-left", "shoulder-right", "head-left", "head-right", "chin-left", "chin-right"]

hands_detected = [0] * 8
hands_world_detected = [0] * 8
targets = [0] * 8

for i in range(8):
    print(f"index {i} is running...")

    # 이미지 파일명 리스트 가져오기
    i_dir = os.path.join(ORIGINAL_DIR, str(i))  # ./upper_pose/original/N
    img_list = [img for img in os.listdir(i_dir) if img.endswith("jpg")]

    # 인덱스별 폴더 생성 : ./upper_pose/pose/N
    # i_dir_pose = os.path.join(POSE_DIR, CLASSES[i])
    # if not os.path.isdir(i_dir_pose):
    #     os.mkdir(i_dir_pose)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for file in tqdm(img_list):
            targets[i] += 1

            read_path = os.path.join(i_dir, file)
            # write_path = os.path.join(i_dir_pose, file)

            # bgr > gray scale
            img_bgr = cv2.imread(read_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # padding
            h, w = img_gray.shape
            dif = w - h
            img = cv2.copyMakeBorder(img_gray, dif // 2, dif // 2, 0, 0, cv2.BORDER_CONSTANT, value=BG_COLOR)

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            results = hands.process(image)

            # Draw face detections of each face.
            if not results.multi_hand_landmarks:
                continue

            hands_detected[i] += 1

            # annotated_image = image.copy()

            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            # annotated_image[:] = BG_COLOR

            # for hand_landmarks in results.multi_hand_landmarks:
            #     print("hand_landmarks:", hand_landmarks)
            #     print(
            #         f"Index finger tip coordinates: (",
            #         f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w}, "
            #         f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h})",
            #     )
            #     mp_drawing.draw_landmarks(
            #         annotated_image,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style(),
            #     )

            # cv2.imwrite("/tmp/annotated_image" + file + ".png", cv2.flip(annotated_image, 1))

            # Draw hand world landmarks.
            if not results.multi_hand_world_landmarks:
                continue

            hands_world_detected[i] += 1

            # for hand_world_landmarks in results.multi_hand_world_landmarks:
            #     mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

            # cv2.imwrite(write_path, annotated_image)

for i in range(8):
    print(f"{CLASSES[i]}: {hands_detected[i]} & {hands_world_detected[i]} ({targets[i]})")
