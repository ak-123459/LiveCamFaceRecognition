import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis


class FaceDetectorAndSaver:
    def __init__(
        self,
        face_model='buffalo_s',
        provider='CPUExecutionProvider',
        use_gpu=False,
        det_size=(640, 640),
        photos_dir=None,
        save_dir=None
    ):
        # Detect project root dynamically
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        # Set photos directory dynamically
        self.photos_dir = photos_dir or os.path.join(self.project_root, "photos")

        # Set output directory dynamically
        self.save_dir = save_dir or os.path.join(self.project_root, "detected_faces")

        self.face_model = face_model
        self.provider = provider
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.app = None

        # Auto-create directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Create image_dict automatically from photos/
        self.image_dict = self.scan_photos()

    def scan_photos(self):
        """
        Scan the photos folder and return dict {image_name_without_ext: path}
        """
        img_dict = {}
        if not os.path.exists(self.photos_dir):
            print(f"[ERROR] Photos directory does not exist: {self.photos_dir}")
            return img_dict

        for file in os.listdir(self.photos_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                uid = os.path.splitext(file)[0]  # filename without extension
                img_path = os.path.join(self.photos_dir, file)
                img_dict[uid] = img_path

        print(f"[INFO] Loaded {len(img_dict)} images from {self.photos_dir}")
        return img_dict

    def init_model(self):
        print("[INFO] Initializing InsightFace model (face detector only)...")
        self.app = FaceAnalysis(name=self.face_model, providers=[self.provider])
        self.app.prepare(ctx_id=0 if self.use_gpu else -1, det_size=self.det_size)
        print("[INFO] Model ready for face detection.")

    def save_detected_faces(self):
        print("[INFO] Detecting faces and saving cropped images...")

        for uid, img_path in self.image_dict.items():
            print(f"[INFO] Processing {uid}: {img_path}")

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Cannot read image: {img_path}")
                continue

            try:
                faces = self.app.get(img)

                if len(faces) == 0:
                    print(f"[WARN] No face found in: {img_path}")
                    continue

                # Create UID folder
                user_folder = os.path.join(self.save_dir, uid)
                os.makedirs(user_folder, exist_ok=True)

                # Save detected faces
                for idx, face in enumerate(faces, start=1):
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    cropped_face = img[y1:y2, x1:x2]

                    save_path = os.path.join(user_folder, f"face_{idx:03d}.jpg")
                    cv2.imwrite(save_path, cropped_face)

                print(f"[INFO] Saved {len(faces)} face(s) under {user_folder}")

            except Exception as e:
                print(f"[ERROR] Failed processing {uid}: {e}")

    def run(self):
        self.init_model()
        self.save_detected_faces()
        print("[INFO] Face extraction complete.")


if __name__ == '__main__':
    detector = FaceDetectorAndSaver(
        use_gpu=False,
        provider="CPUExecutionProvider"
    )

    detector.run()
