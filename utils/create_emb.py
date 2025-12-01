import cv2
import numpy as np
import os
from pathlib import Path
from insightface.app import FaceAnalysis

class FaceEmbeddingSaver:
    def __init__(
        self,
        face_model='buffalo_s',
        provider='CPUExecutionProvider',
        use_gpu=False,
        det_size=(640, 640),
        image_dict=None,
        save_dir=None
    ):
        self.face_model = face_model
        self.provider = provider
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.image_dict = image_dict or {}
        # If save_dir is None, create "face_embeddings" inside main_dir
        self.save_dir = Path(save_dir) if save_dir else Path("face_embeddings")
        self.app = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def init_model(self):
        print("[INFO] Initializing InsightFace model...")
        self.app = FaceAnalysis(name=self.face_model, providers=[self.provider])
        self.app.prepare(ctx_id=0 if self.use_gpu else -1, det_size=self.det_size)
        print("[INFO] InsightFace model ready.")

    def save_embeddings(self):
        print("[INFO] Extracting embeddings and saving to folder...")
        for uid, img_path in self.image_dict.items():
            img_path = Path(img_path)
            if not img_path.exists():
                print(f"[WARN] File does not exist: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[ERROR] Failed to read image: {img_path}")
                continue

            try:
                faces = self.app.get(img)
                if len(faces) == 0:
                    print(f"[WARN] No faces found in image: {img_path}")
                    continue

                # Save each face embedding
                for idx, face in enumerate(faces, start=1):
                    embedding = face.normed_embedding
                    if embedding is not None:
                        user_folder = self.save_dir / uid
                        user_folder.mkdir(parents=True, exist_ok=True)
                        save_path = user_folder / f"face_{idx:03d}.npy"
                        np.save(save_path, embedding)
                        print(f"[INFO] Saved embedding: {save_path}")

            except Exception as e:
                print(f"[ERROR] Failed processing {uid}: {e}")

    def run(self):
        self.init_model()
        self.save_embeddings()
        print("[INFO] All embeddings extracted and saved.")


if __name__ == '__main__':
    # Automatically detect main project folder
    main_dir = Path(__file__).parent / "photos"  # photos inside current script folder
    if not main_dir.exists():
        raise FileNotFoundError(f"Photos folder not found: {main_dir}")

    # Create image dictionary dynamically
    image_dict = {}
    for file_path in main_dir.glob("*.*"):
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            uid = file_path.stem
            image_dict[uid] = file_path

    # Save embeddings inside a folder in the same project
    save_dir = Path(__file__).parent / "face_embeddings"

    saver = FaceEmbeddingSaver(
        image_dict=image_dict,
        save_dir=save_dir,
        use_gpu=False,
        provider='CPUExecutionProvider'
    )
    saver.run()
