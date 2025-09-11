import cv2
import numpy as np
import os
import pickle
import time
from insightface.app import FaceAnalysis
import faiss


class ImageFaceRecognizer:
    def __init__(self,
                 image_path,
                 face_model='buffalo_sc',
                 faiss_index_path='faiss_db/face_index.faiss',
                 faiss_metadata_path='faiss_db/face_index_ids.pkl',
                 haar_cascade_path='det_cv2/haarcascade_frontalface_default.xml',
                 threshold=0.5,
                 providers=['CPUExecutionProvider'],
                 use_gpu=False):

        self.image_path = image_path
        self.faiss_index_path = faiss_index_path
        self.faiss_metadata_path = faiss_metadata_path
        self.haar_cascade_path = haar_cascade_path
        self.threshold = threshold

        # Load Haar cascade
        self._load_haar_cascade()

        # Initialize face model
        self._init_face_model(providers, use_gpu, face_model)

        # Load FAISS index and metadata
        self._load_faiss_index()

    def _load_haar_cascade(self):
        pass
        # self.face_cascade = cv2.CascadeClassifier(self.haar_cascade_path)
        # if self.face_cascade.empty():
        #     raise FileNotFoundError(f"[FATAL] Could not load Haar cascade: {self.haar_cascade_path}")
        # print("[INFO] Haar Cascade loaded.")

    def _init_face_model(self, providers, use_gpu, face_model):
        print("[INFO] Initializing InsightFace model...")
        self.app = FaceAnalysis(name=face_model, providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(512, 512))
        print("[INFO] InsightFace model initialized.")

    def _load_faiss_index(self):
        print("[INFO] Loading FAISS index...")
        if not os.path.exists(self.faiss_index_path) or not os.path.exists(self.faiss_metadata_path):
            raise FileNotFoundError("[FATAL] FAISS index or metadata not found.")

        self.faiss_index = faiss.read_index(self.faiss_index_path)
        with open(self.faiss_metadata_path, 'rb') as f:
            self.faiss_ids = pickle.load(f)

        print(f"[INFO] FAISS index loaded with {self.faiss_index.ntotal} entries.")

    def process_image(self):
        # Load the image
        frame = cv2.imread(self.image_path)
        if frame is None:
            raise FileNotFoundError(f"[FATAL] Cannot read image at {self.image_path}")
        print(f"[INFO] Processing image: {self.image_path}")

        total_start = time.perf_counter()

        # Run face detection and embedding
        pred_start = time.perf_counter()
        faces = self.app.get(frame)
        print(f"[INFO] Number of faces detected: {len(faces)}")
        pred_end = time.perf_counter()

        model_pred_time = pred_end - pred_start
        embeddings = []
        valid_faces = []

        for face in faces:
            if face.normed_embedding is not None:
                embeddings.append(face.normed_embedding)
                valid_faces.append(face)

        similarity_search_time = 0
        if embeddings:
            embeddings = np.stack(embeddings).astype('float32')

            search_start = time.perf_counter()
            scores, indices = self.faiss_index.search(embeddings, 1)
            search_end = time.perf_counter()
            similarity_search_time = search_end - search_start

            for i, face in enumerate(valid_faces):
                x1, y1, x2, y2 = map(int, face.bbox)
                score = float(scores[i][0])
                best_index = int(indices[i][0])
                name = self.faiss_ids[best_index] if score >= self.threshold else "Unknown_"

                color = (0, 255, 0) if name != "Unknown_" else (0, 0, 255)
                label = f"{name.split('_')[0]} ({score:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_y = y1 - 10 if y1 - 10 > 20 else y2 + 20
                cv2.putText(frame, label, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        total_end = time.perf_counter()

        print(f"[TIMING] Model prediction time: {model_pred_time*1000:.2f} ms")
        print(f"[TIMING] Similarity search time: {similarity_search_time*1000:.2f} ms")
        print(f"[TIMING] Total image processing time: {(total_end - total_start)*1000:.2f} ms")

        return frame

    def run(self):
        result_frame = self.process_image()
        cv2.imshow("Face Recognition Result", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Or save:
        # cv2.imwrite("output.jpg", result_frame)

