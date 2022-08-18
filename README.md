# Pocus_Preprocessing

상체데이터 CNN 모델을 위한 이미지 처리 작업

| index | pose                   |
| ----- | ---------------------- |
| 0     | 바른 자세              |
| 1     | 거북목 자세            |
| 2     | 비대칭 어깨 - 왼쪽     |
| 3     | 비대칭 어깨 - 오른쪽   |
| 4     | 기울어진 고개 - 왼쪽   |
| 5     | 기울어진 고개 - 오른쪽 |
| 6     | 턱 괴는 자세 - 왼쪽    |
| 7     | 턱 괴는 자세 - 오른쪽  |

---

**실행 순서**

- 실행을 위해 가상환경 생성 후 cv2, matplotlib, tqdm, mediapipe 설치 필요

1. flip_image.py

2. fliped 이미지 폴더 변경

3. preprocessing.py

4. split_data.py
