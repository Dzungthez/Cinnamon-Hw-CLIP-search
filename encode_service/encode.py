from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import faiss
import clip
import numpy as np
from PIL import Image
import io
import json
import os
import base64
import os
import dotenv
from encode_service.database import get_data_paths, build_index, ImageDataset  # Assuming these functions are in database.py
from search_service.search import load_faiss_index, get_similar_images  # Assuming these functions are in search.py

app = FastAPI()

dotenv.load_dotenv()
model_name = os.getenv("MODEL_NAME")
index_path = os.getenv("INDEX_PATH")
image_paths_path = os.getenv("IMAGES_PATH")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load(model_name, device=device)

index, image_paths = load_faiss_index(index_path, image_paths_path)

class EncodeResponse(BaseModel):
    embedding: List[float]
    image_id: Optional[int] = None
    message: Optional[str] = None


@app.post("/encode", response_model=EncodeResponse)
async def encode_image(file: UploadFile = File(...), add_data: bool = False):
    
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    processed_image = clip_preprocess(image).unsqueeze(0).to(device)
    
    # Encode image
    with torch.no_grad():
        embedding = clip_model.encode_image(processed_image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().flatten()
    
    response = EncodeResponse(embedding=embedding.tolist())

    # Optionally add to index
    if add_data:
        if image_paths and file.filename in image_paths:
            response.image_id = image_paths.index(file.filename)
            response.message = "Image already exists in the index."
        else:
            # Add image to index
            index.add(embedding.reshape(1, -1))
            image_paths.append(file.filename)
            with open(image_paths_path, "w") as f:
                json.dump(image_paths, f)
            faiss.write_index(index, index_path)
            response.image_id = len(image_paths) - 1
            response.message = "Image added to the index."
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
