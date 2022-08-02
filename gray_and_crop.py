import os, cv2
from tqdm import tqdm

BASE_DIR = "./upper_pose/"
ORIGINAL_DIR = os.path.join(BASE_DIR, "original")
GRAY_DIR = os.path.join(BASE_DIR, "gray")
if not os.path.isdir(GRAY_DIR):
    os.mkdir(GRAY_DIR)

for i in range(8):
    print(f"index {i} is running...")

    # 이미지 파일명 리스트 가져오기
    i_dir = os.path.join(ORIGINAL_DIR, str(i))  # ./upper_pose/original/N
    img_list = [img for img in os.listdir(i_dir) if img.endswith("jpg")]

    # 인덱스별 폴더 생성 : ./upper_pose/gray/N
    i_dir_gray = os.path.join(GRAY_DIR, str(i))
    if not os.path.isdir(i_dir_gray):
        os.mkdir(i_dir_gray)

    for img in tqdm(img_list):
        read_path = os.path.join(i_dir, img)

        # rgb > gray scale
        img_bgr = cv2.imread(read_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # crop image
        h, w = img_gray.shape
        set_size = min(h, w)
        mid_x, mid_y = w // 2, h // 2
        offset_x, offset_y = set_size // 2, set_size // 2
        crop_img = img_gray[mid_y - offset_y : mid_y + offset_y, mid_x - offset_x : mid_x + offset_x]

        # ./upper_pose/gray/N/file_name.jpg
        write_path = os.path.join(i_dir_gray, img)
        cv2.imwrite(write_path, crop_img)
