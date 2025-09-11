# import cv2
# from batch_face import RetinaFace
#
# def main():
#     detector = RetinaFace()
#
#     cap = cv2.VideoCapture(0)  # Open default camera
#     if not cap.isOpened():
#         print("Error: Cannot open camera")
#         return
#
#     batch_size = 8  # Number of frames processed in one batch
#     frame_batch = []
#
#     max_size = 1080
#     resize = 1
#     threshold = 0.95
#     batch_size_detector = 100  # Detector batch size (usually >= batch_size)
#
#     print("Press 'q' to quit.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Can't receive frame (stream end?). Exiting ...")
#             break
#
#         # Convert frame from BGR to RGB (detector expects RGB)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_batch.append(frame_rgb)
#
#         # When batch is full, run detection
#         if len(frame_batch) == batch_size:
#             all_faces = detector(
#                 frame_batch,
#                 threshold=threshold,
#                 max_size=max_size,
#                 batch_size=batch_size_detector,
#             )
#
#             # Draw detections on each frame in batch
#             for idx, faces in enumerate(all_faces):
#                 img = frame_batch[idx]
#                 img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display
#
#                 for face in faces:
#                     box, kps, score = face
#                     if score < threshold:
#                         continue
#                     x1, y1, x2, y2 = map(int, box)
#                     cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#                     # Draw keypoints
#                     for (x, y) in kps:
#                         cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 0, 255), -1)
#
#                 cv2.imshow('Face Detection Batch', img_bgr)
#
#             frame_batch = []  # Reset batch
#
#         # Quit on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()


import cv2


save_path = "D:\\video_frames\\"

cap = cv2.VideoCapture(0)  # Open default camera
if not cap.isOpened():
        print("Error: Cannot open camera")

frame_count = 1463
while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imwrite(save_path+str(frame_count)+".jpg",frame)

        frame_count+=1

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()