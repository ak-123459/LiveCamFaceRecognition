import cv2
import numpy as np
import pickle
import os
import faiss
from insightface.app import FaceAnalysis


class FaceEmbeddingIndexer:
    def __init__(
        self,
        face_model='buffalo_s',
        provider='CPUExecutionProvider',
        use_gpu=False,
        det_size=(640, 640),
        faiss_index_path='faiss_db/face_index.faiss',
        faiss_metadata_path='faiss_db/face_index_ids.pkl',
        image_dict=None
    ):
        self.face_model = face_model
        self.provider = provider
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.faiss_index_path = faiss_index_path
        self.faiss_metadata_path = faiss_metadata_path
        self.face_db = {}
        self.app = None
        self.image_dict = image_dict or {}
        self.index = None
        self.existing_ids = []

    def init_model(self):
        print("[INFO] Initializing InsightFace model...")
        self.app = FaceAnalysis(name=self.face_model, providers=[self.provider])
        self.app.prepare(ctx_id=0 if self.use_gpu else -1, det_size=self.det_size)
        print("[INFO] InsightFace model initialized.")

    def load_images_and_extract_embeddings(self):
        print("[INFO] Starting embedding extraction for known images...")
        for uid, img_path in self.image_dict.items():
            print(f"[INFO] Processing {uid}: {img_path}")
            if not os.path.exists(img_path):
                print(f"[WARN] File does not exist: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Failed to read image: {img_path}")
                continue

            try:
                faces = self.app.get(img)
                if len(faces) == 0:
                    print(f"[WARN] No faces found in image: {img_path}")
                    continue

                embedding = faces[0].normed_embedding
                if isinstance(embedding, np.ndarray) and embedding.ndim == 1:
                    self.face_db[uid] = embedding
                    print(f"[INFO] Stored embedding for: {uid}")
                else:
                    print(f"[WARN] Invalid embedding for: {uid}")
            except Exception as e:
                print(f"[ERROR] Failed to extract embedding for {uid}: {e}")

    def load_or_initialize_faiss_index(self):

        if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_metadata_path):
            try:
                print("[INFO] Loading existing FAISS index...")
                self.index = faiss.read_index(self.faiss_index_path)
                with open(self.faiss_metadata_path, 'rb') as f:
                    self.existing_ids = pickle.load(f)
                print(f"[INFO] FAISS index loaded with {self.index.ntotal} vectors.")
                if self.index.ntotal != len(self.existing_ids):
                    print("[WARN] FAISS index and ID metadata mismatch.")
            except Exception as e:
                print(f"[ERROR] Could not load FAISS index: {e}")
                self.index = None
        else:
            print("[INFO] No existing FAISS index found. Creating new one.")
            self.index = None

        if self.index is None:
            self.index = faiss.IndexFlatIP(512)  # Cosine similarity with normalized vectors




    def update_and_save_index(self):

        if not self.face_db:
            print("[WARN] No embeddings to add. Skipping FAISS update.")
            return

        embeddings = np.stack(list(self.face_db.values())).astype('float32')
        ids = list(self.face_db.keys())

        self.index.add(embeddings)
        all_ids = self.existing_ids + ids

        faiss.write_index(self.index, self.faiss_index_path)
        with open(self.faiss_metadata_path, 'wb') as f:
            pickle.dump(all_ids, f)

        print(f"[INFO] Added {len(ids)} embeddings. FAISS index now contains {self.index.ntotal} vectors.")



    def create_index_if_not_exists(self):
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_metadata_path):
            print("[INFO] FAISS index already exists. Skipping creation.")
            return

        print("[INFO] FAISS index not found. Creating new index and embedding database...")
        self.init_model()
        self.load_images_and_extract_embeddings()

        if not self.face_db:
            print("[ERROR] No embeddings were extracted. Cannot create FAISS index.")
            return

        self.index = faiss.IndexFlatIP(512)  # Create new FAISS index
        embeddings = np.stack(list(self.face_db.values())).astype('float32')
        ids = list(self.face_db.keys())

        self.index.add(embeddings)

        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)

        faiss.write_index(self.index, self.faiss_index_path)
        with open(self.faiss_metadata_path, 'wb') as f:
            pickle.dump(ids, f)

        print(f"[INFO] New FAISS index created and saved with {len(ids)} embeddings.")


    def run(self):
        if not (os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_metadata_path)):

            self.create_index_if_not_exists()

        else:
            self.init_model()
            self.load_images_and_extract_embeddings()
            self.load_or_initialize_faiss_index()
            self.update_and_save_index()

        print("[INFO] Face embedding indexing complete.")


if __name__ == '__main__':
    # Example known faces

    path = "C:\\Users\\techma\PycharmProjects\\LiveCamFaceRecognition\\photos\\"
    known_faces_data = {

        "akash_001":path+ "akash_001.jpg",
        "akash_002": path+"akash_002.jpg",
        "akash_003": path+"akash_003.jpg",
        "akash_004": path+"akash_004.jpg",
        "akash_005": path+"akash_005.jpg",


        "kundan_001": path+"kundan_001.jpg",
        "kundan_002": path+"kundan_002.jpg",
        "kundan_003":path+ "kundan_003.jpg",
        "kundan_004": path+"kundan_004.jpg",
        "kundan_005": path+"kundan_005.jpg",

        "aman_001": path+"aman_001.jpg",
        "aman_002": path+"aman_002.jpg",
        "aman_003": path+"aman_002.jpg",
        "aman_004": path+"aman_004.jpg",
        "aman_005": path+"aman_005.jpg",
        "aman_006": path+"aman_006.jpg",
        "aman_007": path+"aman_007.jpg",

    }

    indexer = FaceEmbeddingIndexer(
        image_dict=known_faces_data,faiss_index_path="C:\\Users\\techma\PycharmProjects\LiveCamFaceRecognition\\faiss_db\\face_index.faiss",faiss_metadata_path="C:\\Users\\techma\\PycharmProjects\\LiveCamFaceRecognition\\faiss_db\\face_index_ids.pkl",
        use_gpu=False,  # Set to True if using CUDA
        provider='CPUExecutionProvider'  # Use 'CUDAExecutionProvider' if on GPU
    )

    indexer.run()
