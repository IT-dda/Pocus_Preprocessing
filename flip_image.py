import os, cv2
from tqdm import tqdm

BASE_DIR = "./upper_pose/original"

for i in range(8):
    print(f"index {i} is running...")

    # 이미지 파일명 리스트 가져오기
    img_dir = os.path.join(BASE_DIR, str(i))  # ./upper_pose/original/N
    img_list = [img for img in os.listdir(img_dir) if img.endswith("jpg")]

    # ./upper_pose/original/N/fliped 폴더 생성
    flip_dir = os.path.join(img_dir, "fliped")
    if not os.path.isdir(flip_dir):
        os.mkdir(flip_dir)

    for img in tqdm(img_list):
        # ./upper_pose/original/N/original_file_name.jpg
        read_path = os.path.join(img_dir, img)

        # opencv는 rgb 사용, 사람이 볼 때는 bgr 사용
        img_bgr = cv2.imread(read_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_flip = cv2.flip(img_rgb, 1)  # 1은 좌우반전, 0은 상하반전
        img_flip_bgr = cv2.cvtColor(img_flip, cv2.COLOR_RGB2BGR)

        # ./upper_pose/N/fliped/original_file_name_fliped.jpg
        write_path = os.path.join(flip_dir, f"{img[:-4]}_fliped.jpg")
        cv2.imwrite(write_path, img_flip_bgr)
