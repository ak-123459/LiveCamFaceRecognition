from typing import List

import cv2
import numpy as np
import os
import pickle
import faiss
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from insightface.app import FaceAnalysis

# --- Constants and Configuration ---
# Define common configurations to avoid magic numbers and make them easily adjustable
MODEL_NAME = "buffalo_s"
DET_SIZE = (640, 640)  # A common resolution, can be tuned. Larger is more accurate, smaller is faster.
DEFAULT_THRESHOLD = 0.5
FAISS_DB_DIR = 'faiss_db'
FAISS_INDEX_PATH = os.path.join(FAISS_DB_DIR, 'face_index.faiss')
FAISS_METADATA_PATH = os.path.join(FAISS_DB_DIR, 'face_index_ids.pkl')
HAAR_CASCADE_PATH = 'det_cv2/haarcascade_frontalface_default.xml'
RTSP_URL = 0
MAX_NUM_FACES_TO_DETECT = 5  # Limit the number of faces InsightFace attempts to find per frame





class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super(FrameGrabber, self).__init__()
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.frame_grab_time = 0
        self.frame_count = 0
        self.daemon = True  # Allow the main program to exit even if this thread is still running

    def run(self):
        while not self.stopped:
            start = time.perf_counter()
            ret, frame = self.cap.read()
            end = time.perf_counter()

            if not ret:
                # If frame read fails, wait briefly before retrying to avoid busy-waiting
                time.sleep(0.001)
                continue

            with self.lock:
                self.frame = frame
            self.frame_grab_time = end - start
            self.frame_count += 1
            # Introduce a small sleep to prevent the grabber from hogging CPU if processing is slow
            # This allows other threads to get CPU time.
            time.sleep(0.001)

    def read(self):
        # Return a copy to prevent external modification while the grabber might be writing
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True


class RealTimeFaceRecognizer:
    def __init__(self,
                 face_model: str = MODEL_NAME,
                 faiss_index_path: str = FAISS_INDEX_PATH,
                 faiss_metadata_path: str = FAISS_METADATA_PATH,
                 haar_cascade_path: str = HAAR_CASCADE_PATH,
                 threshold: float = DEFAULT_THRESHOLD,
                 providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                 # Prefer CUDA, fall back to CPU
                 use_gpu: bool = True,  # Explicitly control GPU usage
                 frame_skip: int = 1,
                 num_worker_threads: int = 4):  # Number of threads for concurrent processing of frames (if needed)

        self.faiss_index_path = faiss_index_path
        self.faiss_metadata_path = faiss_metadata_path
        self.haar_cascade_path = haar_cascade_path
        self.threshold = threshold
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.frame_grabber = None
        self.executor = ThreadPoolExecutor(max_workers=num_worker_threads)  # For potential future concurrent tasks

        # Load Haar cascade (optional, if InsightFace's detector is preferred, this can be removed)
        start = time.perf_counter()
        self._load_haar_cascade()
        end = time.perf_counter()
        print(f"[TIMING] Haar cascade loading time: {(end - start) * 1000:.2f} ms")

        # Initialize face model
        start = time.perf_counter()
        self._init_face_model(providers, use_gpu, face_model)
        end = time.perf_counter()
        print(f"[TIMING] Face model initialization time: {(end - start) * 1000:.2f} ms")

        # Load FAISS index and metadata
        start = time.perf_counter()
        self._load_faiss_index()
        end = time.perf_counter()
        print(f"[TIMING] FAISS index and metadata loading time: {(end - start) * 1000:.2f} ms")

        # Initialize webcam
        self._init_webcam()

    def _load_haar_cascade(self):
        # Haar cascade is generally slower and less accurate than modern CNN detectors like InsightFace's.
        # If InsightFace's detector is sufficient, consider removing this to simplify and speed up.
        # It's kept for now as it was in the original code, but its usage is not in `process_frame`.
        self.face_cascade = cv2.CascadeClassifier(self.haar_cascade_path)
        if self.face_cascade.empty():
            print(
                f"[WARNING] Could not load Haar cascade: {self.haar_cascade_path}. Face detection will rely solely on InsightFace.")
            self.face_cascade = None  # Set to None if not loaded to avoid errors
        else:
            print("[INFO] Haar Cascade loaded.")

    def _init_face_model(self, providers: List[str], use_gpu: bool, face_model: str):
        print("[INFO] Initializing InsightFace model...")
        # Prioritize CUDA if use_gpu is True, otherwise use CPU.
        # Ensure providers list is correctly ordered for preference.
        if use_gpu and 'CUDAExecutionProvider' in providers:
            # Reorder providers to ensure CUDA is first if GPU is intended
            providers = ['CUDAExecutionProvider'] + [p for p in providers if p != 'CUDAExecutionProvider']
        else:
            # If not using GPU or CUDA not in providers, ensure CPU is preferred
            providers = ['CPUExecutionProvider'] + [p for p in providers if p != 'CPUExecutionProvider']

        self.app = FaceAnalysis(name=face_model, providers=providers)
        # Use ctx_id=-1 for CPU, 0 for first GPU
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=DET_SIZE)
        print("[INFO] InsightFace model initialized.")

    def _load_faiss_index(self):
        print("[INFO] Loading FAISS index...")
        if not os.path.exists(self.faiss_index_path) or not os.path.exists(self.faiss_metadata_path):
            # Create dummy FAISS index if not found to allow the app to run without a DB
            print("[WARNING] FAISS index or metadata not found. Initializing empty FAISS index.")
            self.faiss_index = faiss.IndexFlatL2(512)  # Assuming 512-dim embeddings
            self.faiss_ids = []
            return

        self.faiss_index = faiss.read_index(self.faiss_index_path)
        with open(self.faiss_metadata_path, 'rb') as f:
            self.faiss_ids = pickle.load(f)

        print(f"[INFO] FAISS index loaded with {self.faiss_index.ntotal} entries.")

    def _init_webcam(self):
        self.cap = cv2.VideoCapture(RTSP_URL)
        # Adjust buffer size to get more recent frames if possible (reduces latency)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size to 1 frame for lower latency

        # Setting resolution for RTSP streams might not always work directly or efficiently.
        # It's better to configure on the camera side if possible.
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Example: set higher resolution
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv2.CAP_PROP_FPS, 30) # Request desired FPS

        if not self.cap.isOpened():
            raise RuntimeError(f"[FATAL] Could not open video stream from {RTSP_URL}.")

        self.frame_grabber = FrameGrabber(self.cap)
        self.frame_grabber.start()
        print("[INFO] Frame grabber thread started.")

    def process_frame(self, frame: np.ndarray):
        total_start = time.perf_counter()

        # 1. Face Detection and Embedding Extraction
        # Use InsightFace's internal detector directly. Haar cascade is likely obsolete here.
        # Limit `max_num` to process only a few faces if performance is critical
        pred_start = time.perf_counter()
        # Convert BGR to RGB for InsightFace if necessary (InsightFace usually expects RGB)
        # Check if frame is 3-channel (color) before converting
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame  # Assume grayscale or already RGB if not 3-channel BGR

        faces = self.app.get(frame_rgb, max_num=MAX_NUM_FACES_TO_DETECT)
        pred_end = time.perf_counter()
        model_pred_time = pred_end - pred_start

        embeddings_to_search = []
        valid_face_info = []  # Store face objects and their original indices

        for face in faces:
            # Ensure embedding exists and is a NumPy array
            if face.normed_embedding is not None and isinstance(face.normed_embedding, np.ndarray):
                embeddings_to_search.append(face.normed_embedding)
                valid_face_info.append(face)  # Store the entire face object for drawing later

        similarity_search_time = 0
        if embeddings_to_search and self.faiss_index.ntotal > 0:  # Only search if there are embeddings and FAISS is not empty
            embeddings_matrix = np.stack(embeddings_to_search).astype('float32')

            # 2. Similarity Search (FAISS)
            search_start = time.perf_counter()
            # FAISS search returns distances and indices. For cosine similarity, use L2 distance on normalized vectors.
            # InsightFace's normed_embedding is already L2 normalized.
            # Faiss.IndexFlatL2 stores Euclidean squared distance.
            # dist = 2 - 2 * cos_similarity => cos_similarity = 1 - dist / 2
            distances, indices = self.faiss_index.search(embeddings_matrix, 1)  # Search for 1 nearest neighbor
            search_end = time.perf_counter()
            similarity_search_time = search_end - search_start

            for i, face in enumerate(valid_face_info):
                x1, y1, x2, y2 = map(int, face.bbox)
                # Convert FAISS distance (L2 squared) to cosine similarity
                # If using IndexFlatL2 for L2-normalized embeddings:
                # distance = 2 * (1 - cosine_similarity)
                # So, cosine_similarity = 1 - (distance / 2)
                raw_faiss_distance = distances[i][0]
                score = 1 - (raw_faiss_distance / 2.0)  # Cosine similarity

                best_index = int(indices[i][0])

                name = "Unknown"
                if best_index != -1 and score >= self.threshold:  # Check if a valid match was found and meets threshold
                    name = self.faiss_ids[best_index]

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                # Clean up name for display (e.g., "ID_001" -> "ID")
                display_name = name.split('_')[0] if "_" in name else name
                label = f"{display_name} ({score:.2f})"

                # 3. Drawing on Frame (Optimized OpenCV operations)
                # Avoid creating new tuples/lists repeatedly if possible, though minor.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_y = y1 - 10 if y1 - 10 > 20 else y2 + 20
                cv2.putText(frame, label, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif self.faiss_index.ntotal == 0 and embeddings_to_search:
            # If FAISS index is empty but faces are detected, draw "No DB"
            for face in valid_face_info:
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for No DB
                text_y = y1 - 10 if y1 - 10 > 20 else y2 + 20
                cv2.putText(frame, "No DB", (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        total_end = time.perf_counter()
        total_time = total_end - total_start

        # Consolidated timing output
        # Using f-strings for cleaner output
        print(
            f"[TIMING] Model: {model_pred_time * 1000:.2f}ms | Search: {similarity_search_time * 1000:.2f}ms | Total Process: {total_time * 1000:.2f}ms | Frame Grab: {self.frame_grabber.frame_grab_time * 1000:.2f}ms")

        return frame

    def run(self):
        print("[INFO] Starting recognition loop. Press 'q' to quit.")
        last_display_time = time.perf_counter()
        while True:
            # Only read frame if it's time to process one (based on frame_skip)
            if self.frame_count % self.frame_skip == 0:
                frame = self.frame_grabber.read()
                if frame is None:
                    time.sleep(0.005)  # Small sleep if no frame is available yet
                    continue

                processed_frame = self.process_frame(frame)

                # Control display rate to avoid overwhelming the GUI thread or CPU
                current_time = time.perf_counter()
                # Aim for approximately 30 FPS display (1000ms / 30 = ~33ms per frame)
                if (current_time - last_display_time) * 1000 > 30:
                    # Resize only for display, not for processing, to save cycles
                    # display_frame = cv2.resize(processed_frame, (640, 480)) # Example: adjust size for display
                    cv2.imshow("Live Face Recognition", processed_frame)
                    last_display_time = current_time

            # Increment frame count regardless of processing, to correctly handle frame_skip
            self.frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' pressed. Exiting loop.")
                break
            elif key == ord('p'):  # Add a pause/play functionality for debugging
                print("[INFO] Paused. Press 'p' to resume.")
                cv2.waitKey(0)  # Wait indefinitely until another key is pressed

        self.cleanup()

    def cleanup(self):
        if self.frame_grabber:
            self.frame_grabber.stop()
            self.frame_grabber.join()  # Wait for the thread to finish
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=True)  # Ensure all tasks in executor are done
        print("[INFO] Resources released. Program exited.")