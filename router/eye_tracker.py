import cv2
import mediapipe as mp
import numpy as np
import ctypes
import pyautogui
import time
import math

def eye_tracker():
    # 마우스 안전모드 비활성화
    pyautogui.FAILSAFE = False

    # 모니터 해상도 가져오기 (Windows 환경)
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    print(f"내 모니터 해상도: {screen_width} x {screen_height}")

    # 두 개의 웹캠 열기: 내장 캠(인덱스 0)와 외부 캠(인덱스 1)
    cap_notebook = cv2.VideoCapture(0)
    cap_phone = cv2.VideoCapture(1)
    if not cap_notebook.isOpened() or not cap_phone.isOpened():
        print("카메라를 열 수 없습니다.")
        exit()

    # 두 캠 해상도 동일하게 설정 (1280 x 720)
    cap_notebook.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_notebook.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_phone.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_phone.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Mediapipe FaceMesh 초기화 (각각 별도 객체)
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh_internal = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_mesh_external = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 사용할 왼쪽 iris landmark 인덱스 (왼쪽 눈만 사용)
    LEFT_IRIS = [469, 470, 471, 472]

    def get_landmark_coords(landmark, width, height):
        return int(landmark.x * width), int(landmark.y * height)

    def compute_pupil_center(landmarks, frame_width, frame_height):
        xs, ys = [], []
        for idx in LEFT_IRIS:
            x, y = get_landmark_coords(landmark=landmarks[idx], width=frame_width, height=frame_height)
            xs.append(x)
            ys.append(y)
        return (int(np.mean(xs)), int(np.mean(ys)))

    # 모니터 해상도 기준의 캘리브레이션 기준점 정의 (ref_points는 화면 좌표로 지정)
    ref_points = {
        "top_left": (0, 0),
        "top": (screen_width // 2, 0),
        "top_right": (screen_width, 0),
        "left": (0, screen_height // 2),
        "center": (screen_width // 2, screen_height // 2),
        "right": (screen_width, screen_height // 2),
        "bottom_left": (0, screen_height),
        "bottom": (screen_width // 2, screen_height),
        "bottom_right": (screen_width, screen_height)
    }
    order = ["top_left", "top", "top_right", "left", "center", "right", "bottom_left", "bottom", "bottom_right"]

    # 캘리브레이션 시, 각 카메라의 정규화된 눈 좌표를 기록
    gaze_points_int = {}
    gaze_points_ext = {}

    # 전체화면 창 생성 (내장 캠 영상 표시)
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("캘리브레이션을 시작합니다.") 
    print("화면에 나타나는 기준점을 응시한 후, 스페이스바를 눌러 기록하세요.")

    while len(gaze_points_int) < len(order) or len(gaze_points_ext) < len(order):
        ret_int, frame_int = cap_notebook.read()
        ret_ext, frame_ext = cap_phone.read()
        if not ret_int or not ret_ext:
            print("카메라 중 하나에서 프레임을 가져올 수 없습니다.")
            continue

        # 좌우 반전
        frame_int = cv2.flip(frame_int, 1)
        frame_ext = cv2.flip(frame_ext, 1)
        
        # 프레임 크기
        h_int, w_int, _ = frame_int.shape
        h_ext, w_ext, _ = frame_ext.shape
        
        # Mediapipe 처리
        rgb_int = cv2.cvtColor(frame_int, cv2.COLOR_BGR2RGB)
        rgb_ext = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
        
        results_int = face_mesh_internal.process(rgb_int)
        results_ext = face_mesh_external.process(rgb_ext)
        
        norm_int = None
        norm_ext = None
        
        if results_int.multi_face_landmarks:
            landmarks_int = results_int.multi_face_landmarks[0].landmark
            pupil_int = compute_pupil_center(landmarks_int, w_int, h_int)
            norm_int = (pupil_int[0] / w_int, pupil_int[1] / h_int)
            cv2.circle(frame_int, pupil_int, 5, (0,255,0), -1)
        
        if results_ext.multi_face_landmarks:
            landmarks_ext = results_ext.multi_face_landmarks[0].landmark
            pupil_ext = compute_pupil_center(landmarks_ext, w_ext, h_ext)
            norm_ext = (pupil_ext[0] / w_ext, pupil_ext[1] / h_ext)
            cv2.circle(frame_ext, pupil_ext, 5, (0,255,0), -1)
        
        # 현재 진행중인 기준점 설정 (순서대로 진행)
        current_label = order[len(gaze_points_int)]
        target_screen_point = ref_points[current_label]
        
        # 내장 캠 영상 전체화면 표시 (리사이즈)
        frame_int_resized = cv2.resize(frame_int, (screen_width, screen_height))
        
        # 전체 기준점 빨간 점 표시
        for label, point in ref_points.items():
            cv2.circle(frame_int_resized, point, 10, (0, 0, 255), -1)
        
        # 현재 진행중인 기준점 강조
        cv2.circle(frame_int_resized, target_screen_point, 15, (0, 0, 255), 2)
        info_text = f"Look at {current_label}"
        cv2.putText(frame_int_resized, info_text, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Calibration", frame_int_resized)
        
        key = cv2.waitKey(1) & 0xFF
        # 스페이스바 입력 시, 두 캠 모두 검출된 경우에만 기록
        if key == 32:
            if norm_int is not None and norm_ext is not None:
                gaze_points_int[current_label] = norm_int
                gaze_points_ext[current_label] = norm_ext
                print(f"{current_label} 캘리브레이션 완료: 내부 {norm_int}, 외부 {norm_ext}")
                time.sleep(0.5)

    cv2.destroyWindow("Calibration")

    if len(gaze_points_int) == 0 or len(gaze_points_ext) == 0:
        print("캘리브레이션 데이터가 수집되지 않았습니다.")
        cap_notebook.release()
        cap_phone.release()
        exit()

    # 먼저 투시변환(각 카메라 별로) 행렬 계산
    # 내부 카메라: 정규화 좌표 -> 화면 좌표 (ref_points)
    src_points_int = np.array([gaze_points_int[label] for label in order], dtype=np.float32)
    dst_points = np.array([ref_points[label] for label in order], dtype=np.float32)
    H_int, _ = cv2.findHomography(src_points_int, dst_points)

    # 외부 카메라: 정규화 좌표 -> 화면 좌표
    src_points_ext = np.array([gaze_points_ext[label] for label in order], dtype=np.float32)
    H_ext, _ = cv2.findHomography(src_points_ext, dst_points)

    print("투시변환 행렬 계산 완료.")

    # 캘리브레이션 데이터로부터 "투시변환+동적 가중치" 결과를 최종 보정하기 위한 다항 회귀 모델 구성
    # 각 캘리브레이션 포인트마다,
    #  1. 각 카메라의 정규화 좌표를 투시변환 적용하여 초기 화면 좌표 예측
    #  2. 동적 가중치로 결합하여 최종 예측 좌표(final_x, final_y) 산출
    #  3. 이 최종 좌표를 입력 피처로 사용하여, 실제 화면 좌표(ref_points)와의 관계를 다항 회귀로 보정

    calib_pred_points = {}
    for label in order:
        # 내부 카메라 투시변환
        norm_int = gaze_points_int[label]
        p_int = np.array([[[norm_int[0], norm_int[1]]]], dtype=np.float32)
        transformed_int = cv2.perspectiveTransform(p_int, H_int)
        x_int_screen = transformed_int[0][0][0]
        y_int_screen = transformed_int[0][0][1]
        
        # 외부 카메라 투시변환 (여기서는 y 좌표를 활용)
        norm_ext = gaze_points_ext[label]
        p_ext = np.array([[[norm_ext[0], norm_ext[1]]]], dtype=np.float32)
        transformed_ext = cv2.perspectiveTransform(p_ext, H_ext)
        y_ext_screen = transformed_ext[0][0][1]
        
        # 동적 가중치 적용 (x는 내부, y는 두 카메라의 결합)
        avg_y_screen = (y_int_screen + y_ext_screen) / 2.0
        normalized_avg_y = avg_y_screen / screen_height  # 0~1 사이 값
        w_int_y = 0.7 - (0.7 - 0.3) * normalized_avg_y
        w_ext_y = 1 - w_int_y
        final_x = x_int_screen
        final_y = w_int_y * y_int_screen + w_ext_y * y_ext_screen
        
        calib_pred_points[label] = (final_x, final_y)

    # 디자인 매트릭스 구성 (항: 1, x, y, x^2, x*y, y^2)
    A_poly = []
    bx_poly = []
    by_poly = []
    for label in order:
        final_x, final_y = calib_pred_points[label]
        sx, sy = ref_points[label]
        A_poly.append([1, final_x, final_y, final_x**2, final_x*final_y, final_y**2])
        bx_poly.append(sx)
        by_poly.append(sy)
    A_poly = np.array(A_poly)
    bx_poly = np.array(bx_poly)
    by_poly = np.array(by_poly)

    coefs_poly_x, _, _, _ = np.linalg.lstsq(A_poly, bx_poly, rcond=None)
    coefs_poly_y, _, _, _ = np.linalg.lstsq(A_poly, by_poly, rcond=None)

    print("최종 다항 회귀 보정 모델 구성 완료. 계수:")
    print("coefs_poly_x:", coefs_poly_x)
    print("coefs_poly_y:", coefs_poly_y)

    click_threshold_seconds = 2.5
    stationary_distance_threshold = 30  # 기존 20에서 30으로 변경
    last_stationary_time = time.time()
    stationary_position = (screen_width / 2, screen_height / 2)
    click_triggered = False

    # 마우스 제어 시작 시, 커서를 모니터 중앙으로 설정
    prev_mouse = (screen_width / 2, screen_height / 2)
    pyautogui.moveTo(int(prev_mouse[0]), int(prev_mouse[1]))

    print("Tracking started. Press 'q' or ESC to quit.")
    smoothing_factor = 0.1  # 지수 스무딩 계수

    while True:
        ret_int, frame_int = cap_notebook.read()
        ret_ext, frame_ext = cap_phone.read()
        if not ret_int or not ret_ext:
            print("Tracking 중 카메라 프레임이 누락되었습니다.")
            break

        frame_int = cv2.flip(frame_int, 1)
        frame_ext = cv2.flip(frame_ext, 1)
        
        h_int, w_int, _ = frame_int.shape
        h_ext, w_ext, _ = frame_ext.shape
        
        rgb_int = cv2.cvtColor(frame_int, cv2.COLOR_BGR2RGB)
        rgb_ext = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB)
        
        results_int = face_mesh_internal.process(rgb_int)
        results_ext = face_mesh_external.process(rgb_ext)
        
        norm_int = None
        norm_ext = None
        if results_int.multi_face_landmarks:
            landmarks_int = results_int.multi_face_landmarks[0].landmark
            pupil_int = compute_pupil_center(landmarks_int, w_int, h_int)
            norm_int = (pupil_int[0] / w_int, pupil_int[1] / h_int)
        if results_ext.multi_face_landmarks:
            landmarks_ext = results_ext.multi_face_landmarks[0].landmark
            pupil_ext = compute_pupil_center(landmarks_ext, w_ext, h_ext)
            norm_ext = (pupil_ext[0] / w_ext, pupil_ext[1] / h_ext)
        
        # 두 캠 모두 검출되어야 처리
        if norm_int is None or norm_ext is None:
            continue

        # 1. 각 카메라의 투시변환 결과 계산
        p_int = np.array([[[norm_int[0], norm_int[1]]]], dtype=np.float32)
        transformed_int = cv2.perspectiveTransform(p_int, H_int)
        x_int_screen = transformed_int[0][0][0]
        y_int_screen = transformed_int[0][0][1]
        
        p_ext = np.array([[[norm_ext[0], norm_ext[1]]]], dtype=np.float32)
        transformed_ext = cv2.perspectiveTransform(p_ext, H_ext)
        y_ext_screen = transformed_ext[0][0][1]
        
        # 2. 동적 가중치 적용하여 결합 (x는 내부, y는 두 캠 결합)
        avg_y_screen = (y_int_screen + y_ext_screen) / 2.0
        normalized_avg_y = avg_y_screen / screen_height
        w_int_y = 0.7 - (0.7 - 0.3) * normalized_avg_y
        w_ext_y = 1 - w_int_y
        initial_x = x_int_screen
        initial_y = w_int_y * y_int_screen + w_ext_y * y_ext_screen

        # 3. 다항 회귀 보정 적용
        features = np.array([1, initial_x, initial_y, initial_x**2, initial_x*initial_y, initial_y**2])
        refined_x = np.dot(coefs_poly_x, features)
        refined_y = np.dot(coefs_poly_y, features)
        
        # 지수 스무딩 적용
        smoothed_mouse_x = prev_mouse[0] + smoothing_factor * (refined_x - prev_mouse[0])
        smoothed_mouse_y = prev_mouse[1] + smoothing_factor * (refined_y - prev_mouse[1])
        pyautogui.moveTo(int(smoothed_mouse_x), int(smoothed_mouse_y))
        prev_mouse = (smoothed_mouse_x, smoothed_mouse_y)
        
        # 클릭 감지 로직
        current_position = (smoothed_mouse_x, smoothed_mouse_y)
        distance = math.hypot(current_position[0] - stationary_position[0],
                            current_position[1] - stationary_position[1])
        
        if distance < stationary_distance_threshold:
            if time.time() - last_stationary_time > click_threshold_seconds and not click_triggered:
                pyautogui.click()
                print("Click!")
                click_triggered = True
        else:
            stationary_position = current_position
            last_stationary_time = time.time()
            click_triggered = False
        
        _, buffer = cv2.imencode('.jpg', frame_int)
        frame_bytes = buffer.tobytes()

        # multipart/x-mixed-replace 방식으로 이미지 스트리밍
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap_notebook.release()
    cap_phone.release()
    face_mesh_internal.close()
    face_mesh_external.close()
    cv2.destroyAllWindows()
