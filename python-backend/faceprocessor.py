
import os
import cv2
import numpy as np
import openface
from typing import Dict, List, Optional, Union, Tuple
from pymongo import MongoClient
import logging

class FaceProcessor:
    def __init__(self, threshold=0.6):
        """Initialize face processor with OpenFace model"""
        self.threshold = threshold  # Threshold for face recognition
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("FaceProcessor")
        
        # Load OpenFace models
        self.logger.info("Loading OpenFace models...")
        
        # Path to the models directory
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'openface', 'models')
        
        # Initialize face detector (Haar Cascade as fallback)
        haar_cascade_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                        'haarcascade', 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        
        # Initialize face alignment
        landmarks_model = os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")
        self.align = openface.AlignDlib(landmarks_model)
        
        # Initialize neural network
        nn_model = os.path.join(model_dir, "nn4.small2.v1.t7")
        self.net = openface.TorchNeuralNet(nn_model, imgDim=96)
        
        # Connect to MongoDB
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["face_recognition_db"]
        self.embeddings_collection = self.db["face_embeddings"]
        
        self.logger.info("Face processor initialized successfully")
    
    def preprocess_image(self, img_data: bytes) -> np.ndarray:
        """Convert image data to OpenCV format and convert to RGB"""
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def detect_face(self, rgb_img: np.ndarray) -> Optional[Tuple]:
        """Detect largest face in image using OpenFace AlignDlib"""
        try:
            # Try to detect using OpenFace's AlignDlib
            bb = self.align.getLargestFaceBoundingBox(rgb_img)
            if bb is not None:
                return (bb.left(), bb.top(), bb.width(), bb.height())
            
            # Fallback to Haar Cascade if AlignDlib fails
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                # Get the largest face
                areas = [w*h for (x, y, w, h) in faces]
                max_idx = np.argmax(areas)
                x, y, w, h = faces[max_idx]
                return (x, y, w, h)
            
            return None
        except Exception as e:
            self.logger.error(f"Error in face detection: {str(e)}")
            return None
    
    def get_face_embedding(self, img_data: bytes) -> Optional[np.ndarray]:
        """Process image and return face embedding vector"""
        try:
            # Preprocess image
            rgb_img = self.preprocess_image(img_data)
            
            # Detect face
            face_rect = self.detect_face(rgb_img)
            if face_rect is None:
                self.logger.warning("No face detected")
                return None
            
            x, y, w, h = face_rect
            
            # Create a bounding box for OpenFace
            bb = openface.AlignDlib.BoundingBox(rect=face_rect, bottom=y+h, right=x+w, top=y, left=x)
            
            # Align face
            aligned_face = self.align.align(96, rgb_img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            
            if aligned_face is None:
                self.logger.warning("Face alignment failed")
                return None
                
            # Get face embedding
            embedding = self.net.forward(aligned_face)
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error in getting face embedding: {str(e)}")
            return None
    
    def save_embedding(self, user_id: str, embedding: np.ndarray, name: Optional[str] = None) -> bool:
        """Save face embedding to database"""
        try:
            # Convert numpy array to list for MongoDB storage
            embedding_list = embedding.tolist()
            
            # Create document
            doc = {
                "user_id": user_id,
                "embedding": embedding_list,
            }
            
            if name:
                doc["name"] = name
                
            # Check if user already exists
            existing = self.embeddings_collection.find_one({"user_id": user_id})
            if existing:
                # Update existing document
                self.embeddings_collection.update_one(
                    {"user_id": user_id},
                    {"$set": doc}
                )
            else:
                # Insert new document
                self.embeddings_collection.insert_one(doc)
                
            return True
        except Exception as e:
            self.logger.error(f"Error saving embedding: {str(e)}")
            return False
    
    def match_face(self, img_data: bytes) -> Optional[Dict]:
        """Match face against database of registered faces"""
        try:
            # Get embedding for the input face
            embedding = self.get_face_embedding(img_data)
            
            if embedding is None:
                return None
                
            # Convert to list for comparison
            embedding_list = embedding.tolist()
            
            # Get all embeddings from database
            all_embeddings = list(self.embeddings_collection.find())
            
            if not all_embeddings:
                return {"matched": False, "message": "No faces registered in database"}
                
            # Find best match
            best_match = None
            best_distance = float('inf')
            
            for doc in all_embeddings:
                stored_embedding = np.array(doc["embedding"])
                
                # Calculate Euclidean distance
                distance = np.sqrt(np.sum(np.square(embedding - stored_embedding)))
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = doc
            
            # Check if the best match is below threshold
            if best_distance < self.threshold:
                return {
                    "matched": True,
                    "user_id": best_match["user_id"],
                    "name": best_match.get("name"),
                    "confidence": (1 - best_distance) * 100  # Convert to percentage
                }
            else:
                return {
                    "matched": False,
                    "confidence": (1 - best_distance) * 100  # Convert to percentage
                }
                
        except Exception as e:
            self.logger.error(f"Error in matching face: {str(e)}")
            return None
    
    def compare_faces(self, source_img_data: bytes, target_img_data: bytes) -> Optional[Dict]:
        """Compare two faces directly without using the database"""
        try:
            # Get embeddings for both faces
            source_embedding = self.get_face_embedding(source_img_data)
            target_embedding = self.get_face_embedding(target_img_data)
            
            if source_embedding is None or target_embedding is None:
                return None
                
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum(np.square(source_embedding - target_embedding)))
            
            # Check if distance is below threshold
            matched = distance < self.threshold
            confidence = (1 - distance) * 100  # Convert to percentage
            
            return {
                "matched": matched,
                "confidence": confidence
            }
                
        except Exception as e:
            self.logger.error(f"Error in comparing faces: {str(e)}")
            return None