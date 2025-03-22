
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import base64
from typing import Optional
import os
import json
import numpy as np
from io import BytesIO
from PIL import Image

# Import face processor class
from faceprocessor import FaceProcessor

app = FastAPI(title="Face Recognition API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face processor
face_processor = FaceProcessor()

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.post("/api/extract-embedding")
async def extract_embedding(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    """Extract face embedding from uploaded image"""
    try:
        # Read image file
        contents = await file.read()
        
        # Process image and get embedding
        embedding = face_processor.get_face_embedding(contents)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Return the embedding
        return {
            "success": True,
            "embedding": embedding.tolist(),
            "user_id": user_id
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/register-face")
async def register_face(file: UploadFile = File(...), user_id: str = Form(...), name: Optional[str] = Form(None)):
    """Register a face in the database"""
    try:
        contents = await file.read()
        
        # Process image and get embedding
        embedding = face_processor.get_face_embedding(contents)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Save embedding to database
        success = face_processor.save_embedding(user_id, embedding, name)
        
        return {
            "success": success,
            "message": "Face registered successfully" if success else "Failed to register face"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/match-face")
async def match_face(file: UploadFile = File(...)):
    """Match a face against registered faces"""
    try:
        contents = await file.read()
        
        # Process image and match face
        match_result = face_processor.match_face(contents)
        
        if match_result is None:
            return {
                "success": True,
                "matched": False,
                "message": "No face detected in the image"
            }
            
        return {
            "success": True,
            "matched": match_result["matched"],
            "user_id": match_result.get("user_id"),
            "name": match_result.get("name"),
            "confidence": match_result.get("confidence")
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/match-specific-face")
async def match_specific_face(file: UploadFile = File(...), target_file: UploadFile = File(...)):
    """Match a face from camera against a specific uploaded face"""
    try:
        # Read both image files
        source_contents = await file.read()
        target_contents = await target_file.read()
        
        # Compare the two faces directly
        match_result = face_processor.compare_faces(source_contents, target_contents)
        
        if match_result is None:
            return {
                "success": True,
                "matched": False,
                "message": "Face detection failed in one or both images"
            }
            
        return {
            "success": True,
            "matched": match_result["matched"],
            "confidence": match_result["confidence"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)