# import cv2
# import threading
# import time
# import numpy as np
# from ultralytics import YOLO
#
# # -------------------- Configuration --------------------
#
# RTSP_URL = "rtsp://admin:tdbtech4189@192.168.1.253:554"  # Use 0 for webcam, or replace with RTSP stream
# MODEL_PATH = '../yolov8m.pt'
# TARGET_CLASS = "cell phone"
#
# TARGET_STATS = {
#     "brightness": 85.54,
#     "contrast": 100.13,
#     "saturation": 20.27,
#     "temperature": 5.18,
#     "bgr": np.array([106, 109, 109])
# }
#
#
#
# # Adjustment limits
# ADJUST_LIMITS = {
#     "brightness_shift": (-50, 50),
#     "contrast_factor": (0.5, 2.0),
#     "saturation_factor": (0.5, 2.0)
# }
#
# # -------------------- Globals --------------------
#
# frame = None
# lock = threading.Lock()
# running = True
#
# # -------------------- Functions --------------------
#
# def compute_stats(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     brightness = np.mean(gray)
#     contrast = np.std(gray)
#     saturation = np.mean(hsv[:, :, 1])
#     avg_color = np.mean(img, axis=(0, 1))
#     red, green, blue = avg_color[2], avg_color[1], avg_color[0]
#     temperature = red - blue
#     return brightness, contrast, saturation, temperature, avg_color.astype(int)
#
#
# def adjust_image(img, brightness=0, contrast=1.0, saturation=1.0):
#     img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
#     hsv[:, :, 1] *= saturation
#     hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
#     img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
#     return img
#
#
# def calculate_adjustments(current_stats, target_stats):
#     brightness_adj = target_stats["brightness"] - current_stats[0]
#     contrast_adj = target_stats["contrast"] / (current_stats[1] + 1e-5)
#     saturation_adj = target_stats["saturation"] / (current_stats[2] + 1e-5)
#
#     # Clip to safe ranges
#     brightness_adj = np.clip(brightness_adj, *ADJUST_LIMITS["brightness_shift"])
#     contrast_adj = np.clip(contrast_adj, *ADJUST_LIMITS["contrast_factor"])
#     saturation_adj = np.clip(saturation_adj, *ADJUST_LIMITS["saturation_factor"])
#
#     return brightness_adj, contrast_adj, saturation_adj
#
#
# def display_image_info(stats):
#     brightness, contrast, saturation, temperature, avg_bgr = stats
#     print("\n📷 Image Description:")
#     print(f"💡 Brightness: {brightness:.2f}")
#     print(f"🎨 Contrast: {contrast:.2f}")
#     print(f"🌈 Saturation: {saturation:.2f}")
#     print(f"🔥 Color Temp (R-B): {temperature:.2f}")
#     print(f"🟢 Avg BGR: {avg_bgr}")
#
#
# def draw_boxes(frame, results, labels_map):
#     for box in results.boxes:
#         cls_id = int(box.cls[0])
#         label = labels_map[cls_id]
#         conf = float(box.conf[0])
#
#         if label != TARGET_CLASS:
#             continue
#
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     return frame
#
#
# def capture_frames():
#     global frame, running
#     cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         print("❌ Failed to open video stream.")
#         running = False
#         return
#
#     while running:
#         ret, new_frame = cap.read()
#         if not ret:
#             print("⚠️ Failed to read frame.")
#             break
#         with lock:
#             frame = new_frame
#
#     cap.release()
#
# # -------------------- Main Loop --------------------
#
# def main():
#
#
#     global running
#     model = YOLO(MODEL_PATH)
#     print("✅ Model loaded. Classes:", model.names)
#
#     thread = threading.Thread(target=capture_frames)
#     thread.start()
#
#     while running:
#         start_time = time.time()
#
#         with lock:
#             if frame is None:
#                 continue
#             raw_frame = frame.copy()
#
#         # 1. Compute current image stats
#         stats = compute_stats(raw_frame)
#
#         # 2. Compute adjustments
#         b_shift, c_factor, s_factor = calculate_adjustments(stats, TARGET_STATS)
#
#         # 3. Apply adjustments
#         adjusted_frame = adjust_image(raw_frame, brightness=b_shift,
#                                       contrast=c_factor, saturation=s_factor)
#
#         # 4. Display stats
#         print("🔧 Adjustments: "
#               f"Brightness={b_shift:.1f}, Contrast={c_factor:.2f}, Saturation={s_factor:.2f}")
#         display_image_info(stats)
#
#         # 5. Run YOLO inference
#         results = model(adjusted_frame, conf=0.25)[0]
#
#         # 6. Draw detections
#         display_frame = draw_boxes(adjusted_frame, results, model.names)
#
#         # 7. Show FPS
#         fps = 1 / (time.time() - start_time)
#         cv2.putText(display_frame, f"FPS: {fps:.2f}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#
#         # 8. Show final output
#         cv2.imshow("Detection & Adjusted Frame", display_frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # ESC
#             running = False
#             break
#
#     cv2.destroyAllWindows()
#     thread.join()
# #
# # # -------------------- Entry Point --------------------
# #
# # if __name__ == "__main__":
# #     main()




#  -------------------------------------------------------------------------------------------------------



# pip install insightface
# pip install onnxruntime

#
# import cv2
# from insightface.app import FaceAnalysis
# import os
#
#
# app = FaceAnalysis(name="buffalo_s")
# app.prepare(ctx_id=0)  # Use 0 for CPU, >0 for GPU
#
# img_path = "D:\\video_frames\\"
#
# for img_name in os.listdir(img_path):
#
#         img = cv2.imread(img_path+img_name)
#
#         print("image name is :-> ",img_name)
#         faces = app.get(img)
#
#         if faces:
#             face = faces[0]
#             print("Pose (yaw, pitch, roll):", face.pose)  # Direct face angle
#             print("Bounding Box:", face.bbox)
#
#             # Draw bounding box
#             x1, y1, x2, y2 = map(int, face.bbox)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#             cv2.putText(img, f"Yaw: {face.pose[0]:.2f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#             cv2.putText(img, f"Pitch: {face.pose[1]:.2f}", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#             cv2.putText(img, f"Roll: {face.pose[2]:.2f}", (10, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#             cv2.imshow("Detected Face", img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         else:
#             print("No face detected.")
#
#
