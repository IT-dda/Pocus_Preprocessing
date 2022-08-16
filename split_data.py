import os, shutil, random

BASE_DIR = "./upper_pose/"

# train, validation, test 폴더 생성
# ./upper_pose/train
train_dir = os.path.join(BASE_DIR, "train")
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

# ./upper_pose/validation
# validation_dir = os.path.join(BASE_DIR, "validation")
# if not os.path.isdir(validation_dir):
#     os.mkdir(validation_dir)

# ./upper_pose/test
test_dir = os.path.join(BASE_DIR, "test")
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)


# 각 index 폴더 생성
# ./upper_pose/[train|validation|test]/N
for i in range(8):
    train_i_dir = os.path.join(train_dir, str(i))
    if not os.path.isdir(train_i_dir):
        os.mkdir(train_i_dir)

    # validation_i_dir = os.path.join(validation_dir, str(i))
    # if not os.path.isdir(validation_i_dir):
    #     os.mkdir(validation_i_dir)

    test_i_dir = os.path.join(test_dir, str(i))
    if not os.path.isdir(test_i_dir):
        os.mkdir(test_i_dir)


gray_dir = os.path.join(BASE_DIR, "face")

# 파일명 가져오기
fnames = []
for i in range(8):
    dir_i = os.path.join(gray_dir, str(i))
    img_list = [img for img in os.listdir(dir_i) if img.endswith("jpg")]
    random.shuffle(img_list)  # 랜덤으로 파일 나누도록
    fnames.append(img_list)

# 파일 분류
for idx, fname in enumerate(fnames):
    i_dir = os.path.join(gray_dir, str(idx))
    i_dir_train = os.path.join(train_dir, str(idx))
    # i_dir_validation = os.path.join(validation_dir, str(idx))
    i_dir_test = os.path.join(test_dir, str(idx))

    # 비율 설정
    l = len(fname)
    train_rate = int(l * 0.8)
    # validation_rate = train_rate + int(len(fname) * 0.2)
    # test_rate = len(fname)

    print(f"index {idx} - 전체 파일: {l} // train: {train_rate} // test: {l-train_rate}")

    # train
    train_data = fname[:train_rate]
    for f in train_data:
        src = os.path.join(i_dir, f)
        dst = os.path.join(i_dir_train, f)
        shutil.copyfile(src, dst)

    # validation
    # validation_data = fname[train_rate:validation_rate]
    # for f in validation_data:
    #     src = os.path.join(i_dir, f)
    #     dst = os.path.join(i_dir_validation, f)
    #     shutil.copyfile(src, dst)

    # test
    test_data = fname[train_rate:]
    for f in test_data:
        src = os.path.join(i_dir, f)
        dst = os.path.join(i_dir_test, f)
        shutil.copyfile(src, dst)
