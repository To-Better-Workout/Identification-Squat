import cv2
import mediapipe as mp
import numpy as np
import time
from insightface.app import FaceAnalysis

# --------------------------
# 1. FaceRecognizer Class
# --------------------------
class FaceRecognizer:
    def __init__(self, sim_threshold=0.3, countdown_duration=10, capture_duration=3, grace_period=2.0):
        """
        :param sim_threshold: Cosine similarity threshold. Above => Same person
        :param countdown_duration: Seconds before capture starts
        :param capture_duration: Seconds to pick the best face
        :param grace_period: Keep recognized = True for N seconds after last valid face match
        """
        # MediaPipe FaceDetection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.2
        )

        # InsightFace
        self.app = FaceAnalysis(
            allowed_modules=['detection', 'recognition'],
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(224, 224))

        # Similarity threshold
        self.SIM_THRESHOLD = sim_threshold

        # Registration timers/flags
        self.countdown_duration = countdown_duration
        self.capture_duration = capture_duration
        self.countdown_start_time = time.time()
        self.countdown_done = False
        self.capturing = False
        self.capture_start_time = None

        # Store the final embedding
        self.registered_embedding = None
        # Track largest face during 3s capture
        self.best_face_area = 0
        self.best_face_emb = None

        # For short-term "memory" of bounding box (optional)
        self.last_box = None
        self.frames_since_last_detection = 0
        self.MAX_NO_DETECT_FRAMES = 10

        # GRACE PERIOD logic:
        self.grace_period = grace_period  # seconds
        self.recognized_state = False
        self.last_recognized_time = 0.0

        # We'll store the bounding box of the recognized face
        self.recognized_box = None  # (x_min, y_min, w, h)

    def get_embedding(self, face_img_rgb):
        """
        Convert face region (RGB) to BGR, run through InsightFace, return embedding array.
        """
        face_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(face_bgr)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    def compute_cosine_similarity(self, emb1, emb2):
        """
        Cosine similarity (closer to 1 => more similar).
        """
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1_norm, emb2_norm)

    def update(self, frame):
        """
        Pass in a BGR frame, handle face registration or face recognition.
        Returns:
            annotated_frame (with bounding boxes / text),
            recognized_state (bool) -> True if we consider the *registered person* present,
            recognized_box (tuple) -> bounding box (x_min, y_min, w, h) of recognized face (or None).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        now = time.time()

        # Default: no recognized face *this frame*
        recognized_in_this_frame = False
        recognized_box_this_frame = None

        # -------------------------------
        # A) Countdown Phase
        # -------------------------------
        if not self.countdown_done and self.registered_embedding is None:
            elapsed = now - self.countdown_start_time
            remain = int(self.countdown_duration - elapsed)
            if remain > 0:
                text = f"Get ready! Registration in {remain} s"
                font_scale = 1.2
                thickness = 3
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                      font_scale, thickness)
                x_pos = (w - text_w) // 2
                y_pos = (h + text_h) // 2
                cv2.putText(frame, text, (x_pos, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255),
                            thickness, cv2.LINE_AA)
            else:
                # Countdown finished -> start capturing
                self.countdown_done = True
                self.capturing = True
                self.capture_start_time = time.time()

        # -------------------------------
        # B) Face Capture Phase (3s)
        # -------------------------------
        elif self.capturing and self.registered_embedding is None:
            capture_elapsed = now - self.capture_start_time
            remain_cap = int(self.capture_duration - capture_elapsed)

            if remain_cap > 0:
                text = f"Capturing face... {remain_cap} s remaining"
                font_scale = 1.0
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                      font_scale, thickness)
                x_pos = (w - text_w) // 2
                y_pos = (h + text_h) // 2
                cv2.putText(frame, text, (x_pos, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255),
                            thickness, cv2.LINE_AA)

                results = self.face_detector.process(frame_rgb)
                if results.detections:
                    self.frames_since_last_detection = 0
                    for detection in results.detections:
                        box = detection.location_data.relative_bounding_box
                        x_min = int(box.xmin * w)
                        y_min = int(box.ymin * h)
                        box_w = int(box.width * w)
                        box_h = int(box.height * h)

                        # Clamp
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        box_w = min(w - x_min, box_w)
                        box_h = min(h - y_min, box_h)

                        area = box_w * box_h
                        # Keep largest face
                        if area > self.best_face_area:
                            face_crop = frame_rgb[y_min:y_min+box_h, x_min:x_min+box_w]
                            emb = self.get_embedding(face_crop)
                            if emb is not None:
                                self.best_face_area = area
                                self.best_face_emb = emb

                        # Draw bounding box
                        cv2.rectangle(frame, (x_min, y_min),
                                      (x_min + box_w, y_min + box_h),
                                      (0, 255, 255), 2)
                        cv2.putText(frame, "Capturing Face",
                                    (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)
                else:
                    self.frames_since_last_detection += 1
            else:
                # Capture done
                self.capturing = False
                if self.best_face_emb is not None:
                    self.registered_embedding = self.best_face_emb
                    print("[INFO] Face registration complete!")
                else:
                    print("[WARN] No valid face found. Registration is None.")

        # -------------------------------
        # C) Normal Operation (Compare)
        # -------------------------------
        else:
            if self.registered_embedding is not None:
                results = self.face_detector.process(frame_rgb)
                if results.detections:
                    self.frames_since_last_detection = 0
                    for detection in results.detections:
                        box = detection.location_data.relative_bounding_box
                        x_min = int(box.xmin * w)
                        y_min = int(box.ymin * h)
                        box_w = int(box.width * w)
                        box_h = int(box.height * h)

                        # Clamp
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        box_w = min(w - x_min, box_w)
                        box_h = min(h - y_min, box_h)

                        self.last_box = (x_min, y_min, box_w, box_h)

                        # Compare embeddings
                        face_crop = frame_rgb[y_min:y_min+box_h, x_min:x_min+box_w]
                        emb = self.get_embedding(face_crop)
                        if emb is not None:
                            sim = self.compute_cosine_similarity(self.registered_embedding, emb)
                            if sim > self.SIM_THRESHOLD:
                                recognized_in_this_frame = True
                                recognized_box_this_frame = (x_min, y_min, box_w, box_h)

                                color = (0, 255, 0)
                                label = f"Same Person (sim={sim:.2f})"
                            else:
                                color = (0, 0, 255)
                                label = f"Different (sim={sim:.2f})"

                            cv2.rectangle(frame, (x_min, y_min),
                                          (x_min + box_w, y_min + box_h),
                                          color, 2)
                            cv2.putText(frame, label,
                                        (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        color, 2)
                else:
                    self.frames_since_last_detection += 1
                    # Optionally, draw last known bounding box for a while
                    if (self.last_box is not None and
                            self.frames_since_last_detection < self.MAX_NO_DETECT_FRAMES):
                        (x_min, y_min, box_w, box_h) = self.last_box
                        color = (0, 255, 255)
                        label = "Last Known Face"
                        cv2.rectangle(frame, (x_min, y_min),
                                      (x_min + box_w, y_min + box_h),
                                      color, 2)
                        cv2.putText(frame, label, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        self.last_box = None

        # ---------------------------
        # D) Grace Period Update
        # ---------------------------
        if recognized_in_this_frame:
            self.recognized_state = True
            self.last_recognized_time = now
            self.recognized_box = recognized_box_this_frame
        else:
            if self.recognized_state and (now - self.last_recognized_time < self.grace_period):
                # keep recognized_state = True for grace_period
                pass
            else:
                self.recognized_state = False
                self.recognized_box = None

        return frame, self.recognized_state, self.recognized_box


# --------------------------
# 2. SquatCounter Class
# --------------------------
class SquatCounter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.squat_count = 0
        self.state = 0  # 0 = Standing, 1 = Squatting

    def calculate_angle(self, a, b, c):
        """
        Helper: Calculate angle between three points (a-b-c) using the dot product.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def update(self, frame, recognized_box=None):
        """
        Run pose estimation on the given frame (BGR).
        Only count if the NOSE is inside recognized_box (i.e., same person).
        Returns the annotated frame.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image_bgr.shape

            # Nose landmark
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            # ----------------------------------------------------
            # 1) Check if the skeleton belongs to recognized person
            # ----------------------------------------------------
            # Only if recognized_box is not None,
            # we check if (nose_x, nose_y) is inside that box
            # recognized_box = (x_min, y_min, box_w, box_h)

            belongs_to_registered_user = False
            if recognized_box is not None:
                (rx, ry, rw, rh) = recognized_box
                rx2 = rx + rw
                ry2 = ry + rh
                # If nose is inside face bounding box => same person
                if (rx <= nose_x <= rx2) and (ry <= nose_y <= ry2):
                    belongs_to_registered_user = True

            # If the skeleton does not match the recognized person, we skip the counting logic
            if not belongs_to_registered_user:
                # Draw the pose as usual, but do NOT do squat counting
                mp.solutions.drawing_utils.draw_landmarks(
                    image_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                cv2.putText(image_bgr, f'Squat Count: {self.squat_count}', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, "Another Person (No Counting)",
                            (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                return image_bgr

            # ----------------------------------------------------
            # 2) If it belongs to recognized user => do squat logic
            # ----------------------------------------------------
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Calculate knee angles
            left_knee_angle = self.calculate_angle(
                [left_hip.x, left_hip.y],
                [left_knee.x, left_knee.y],
                [left_ankle.x, left_ankle.y]
            )
            right_knee_angle = self.calculate_angle(
                [right_hip.x, right_hip.y],
                [right_knee.x, right_knee.y],
                [right_ankle.x, right_ankle.y]
            )

            # Detect transitions
            if self.state == 0 and left_knee_angle >= 170 and right_knee_angle >= 170:
                # Standing
                self.state = 0
            elif self.state == 0 and left_knee_angle <= 125 and right_knee_angle <= 125:
                # Moving into squat
                self.state = 1
            elif self.state == 1 and left_knee_angle >= 170 and right_knee_angle >= 170:
                # Coming back up
                self.state = 0
                self.squat_count += 1
                print(f"Squat Count: {self.squat_count}")

            # Draw the knee angles
            cv2.putText(image_bgr, f'{int(left_knee_angle)} deg',
                        (int(left_knee.x * w), int(left_knee.y * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image_bgr, f'{int(right_knee_angle)} deg',
                        (int(right_knee.x * w), int(right_knee.y * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

            mp.solutions.drawing_utils.draw_landmarks(
                image_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        # Show squat count
        cv2.putText(image_bgr, f'Squat Count: {self.squat_count}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return image_bgr


# --------------------------
# 3. Main / Combined Logic
# --------------------------
def main():
    # A) Initialize classes
    face_recog = FaceRecognizer(sim_threshold=0.3,
                                countdown_duration=10,
                                capture_duration=3,
                                grace_period=2.0)
    squat_counter = SquatCounter()

    # B) Setup camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    cap.set(cv2.CAP_PROP_FPS, 27)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate + flip if needed
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)

        # 1) Face recognition update
        face_frame, is_recognized, recognized_box = face_recog.update(frame)

        # 2) If recognized, do squat logic
        #    Otherwise, just show face detection bounding boxes
        if is_recognized:
            combined_frame = squat_counter.update(face_frame, recognized_box)
        else:
            combined_frame = face_frame

        cv2.imshow("Face & Squat Counter", combined_frame)

        # ESC or 'q' to exit
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
