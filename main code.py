# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_gradient_magnitude

# # -------------------------------
# # MediaPipe Setup
# # -------------------------------
# mp_hands = mp.solutions.hands
# mp_face = mp.solutions.face_mesh

# hands = mp_hands.Hands(max_num_hands=1)
# face_mesh = mp_face.FaceMesh(static_image_mode=False)

# # -------------------------------
# # Physics Parameters
# # -------------------------------
# TOUCH_THRESHOLD_CM = 2.0
# LIGHT_INTENSITY = 1.0  # normalized

# # -------------------------------
# # Shadow Depth Estimation Function
# # -------------------------------
# def estimate_depth_from_shadow(shadow_area, gradient_strength):
#     """
#     Physics-based depth estimation using inverse shadow spread.
#     """
#     if shadow_area < 1:
#         return 10.0  # Far away

#     z_distance = 1 / (shadow_area * gradient_strength + 1e-6)
#     z_cm = np.clip(z_distance * 100, 0, 10)
#     return z_cm

# # -------------------------------
# # Shadow Detection
# # -------------------------------
# def detect_shadow(face_roi):
#     gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (11, 11), 0)

#     # Shadow = lower intensity regions
#     shadow_mask = cv2.adaptiveThreshold(
#         blurred, 255,
#         cv2.ADAPTIVE_THRESH_MEAN_C,
#         cv2.THRESH_BINARY_INV,
#         15, 3
#     )

#     return shadow_mask, gray

# # -------------------------------
# # Heatmap Generation
# # -------------------------------
# def generate_intensity_matrix(gray_face):
#     norm = gray_face / 255.0
#     intensity_loss = 1 - norm
#     return intensity_loss

# # -------------------------------
# # Main Loop
# # -------------------------------
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     hand_results = hands.process(rgb)
#     face_results = face_mesh.process(rgb)

#     action = "No Action"
#     distance_cm = None

#     if face_results.multi_face_landmarks:
#         face_landmarks = face_results.multi_face_landmarks[0]
#         xs = [int(lm.x * w) for lm in face_landmarks.landmark]
#         ys = [int(lm.y * h) for lm in face_landmarks.landmark]

#         x_min, x_max = max(min(xs), 0), min(max(xs), w)
#         y_min, y_max = max(min(ys), 0), min(max(ys), h)

#         face_roi = frame[y_min:y_max, x_min:x_max]

#         if face_roi.size > 0:
#             shadow_mask, gray_face = detect_shadow(face_roi)
#             shadow_area = np.sum(shadow_mask > 0)

#             gradient_strength = np.mean(
#                 gaussian_gradient_magnitude(gray_face, sigma=1)
#             )

#             distance_cm = estimate_depth_from_shadow(
#                 shadow_area, gradient_strength
#             )

#             if distance_cm < TOUCH_THRESHOLD_CM:
#                 action = "Touching Face / Eating"

#             # Visualization
#             heatmap = generate_intensity_matrix(gray_face)
#             heatmap = cv2.applyColorMap(
#                 (heatmap * 255).astype(np.uint8),
#                 cv2.COLORMAP_JET
#             )

#             frame[y_min:y_max, x_min:x_max] = cv2.addWeighted(
#                 face_roi, 0.6, heatmap, 0.4, 0
#             )

#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#     if distance_cm:
#         cv2.putText(
#             frame,
#             f"Distance: {distance_cm:.2f} cm",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 255),
#             2
#         )

#     cv2.putText(
#         frame,
#         f"Action: {action}",
#         (30, 80),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 0, 255),
#         2
#     )

#     cv2.imshow("Physics-Based Shadow Depth Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import mediapipe as mp

# ============================
# MediaPipe Setup
# ============================
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1)
face_mesh = mp_face.FaceMesh(static_image_mode=False)

# ============================
# Parameters
# ============================
TOUCH_THRESHOLD_CM = 2.0
baseline_face_gray = None
baseline_shape = None

# ============================
# Shadow Calculation
# ============================
def calculate_hand_shadow(current_gray, baseline_gray):
    diff = baseline_gray.astype(np.int16) - current_gray.astype(np.int16)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    _, shadow_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    return shadow_mask, diff

# ============================
# Physics-Based Depth Estimation
# ============================
def estimate_depth(shadow_mask):
    shadow_area = np.sum(shadow_mask > 0)

    if shadow_area < 50:
        return 10.0

    edge_strength = np.mean(cv2.Canny(shadow_mask, 50, 150))
    distance = 5 / (shadow_area * edge_strength + 1e-5)
    return np.clip(distance, 0, 10)

# ============================
# Main Loop
# ============================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)

    action = "No Action"
    distance_cm = None

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
        ys = [int(lm.y * h) for lm in face_landmarks.landmark]

        x_min, x_max = max(min(xs), 0), min(max(xs), w)
        y_min, y_max = max(min(ys), 0), min(max(ys), h)

        face_roi = frame[y_min:y_max, x_min:x_max]

        if face_roi.size > 0:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # ----------------------------
            # Baseline capture
            # ----------------------------
            if baseline_face_gray is None:
                baseline_face_gray = gray_face.copy()
                baseline_shape = baseline_face_gray.shape

                cv2.putText(frame,
                            "Calibrating lighting...",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)

            else:
                # Resize current face to baseline size
                current_gray_resized = cv2.resize(
                    gray_face,
                    (baseline_shape[1], baseline_shape[0])
                )

                shadow_mask, shadow_diff = calculate_hand_shadow(
                    current_gray_resized, baseline_face_gray
                )

                distance_cm = estimate_depth(shadow_mask)

                if distance_cm < TOUCH_THRESHOLD_CM:
                    action = "Touching Face / Eating"

                # Visualization
                heatmap = cv2.applyColorMap(shadow_diff, cv2.COLORMAP_JET)
                face_overlay = cv2.addWeighted(
                    cv2.resize(face_roi, (baseline_shape[1], baseline_shape[0])),
                    0.6, heatmap, 0.4, 0
                )

                frame[y_min:y_max, x_min:x_max] = cv2.resize(
                    face_overlay,
                    (x_max - x_min, y_max - y_min)
                )

                # Debug
                cv2.imshow("Shadow Mask", shadow_mask)
                cv2.imshow("Shadow Difference", shadow_diff)

                cv2.rectangle(frame,
                              (x_min, y_min),
                              (x_max, y_max),
                              (0, 255, 0), 2)

    if distance_cm is not None:
        cv2.putText(frame,
                    f"Distance: {distance_cm:.2f} cm",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

    cv2.putText(frame,
                f"Action: {action}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("Physics-Based Shadow Depth Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
