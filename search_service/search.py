from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import faiss
import torch
import clip
import numpy as np
from PIL import Image
import json
from typing import List
import matplotlib.pyplot as plt
import io
import os
import dotenv
import tempfile

dotenv.load_dotenv()

model_name = os.getenv("MODEL_NAME")
index_path = os.getenv("INDEX_PATH")
image_paths = os.getenv("IMAGES_PATH")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = FastAPI()

def load_faiss_index(index_path, image_paths_path):
    index = faiss.read_index(index_path)
    with open(image_paths_path, "r") as f:
        image_paths = json.load(f)
    return index, image_paths

def get_similar_images(images, index, image_paths, top_k=5, model_name=model_name, device="cpu"):
    clip_model, clip_preprocess = clip.load(model_name, device=device)

    processed_images = [clip_preprocess(img).unsqueeze(0).to(device) for img in images]
    processed_images = torch.cat(processed_images, dim=0) 

    with torch.no_grad():
        embeddings = clip_model.encode_image(processed_images)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy().astype('float32')

    D, I = index.search(embeddings, top_k)
    similar_images = [[image_paths[i] for i in indices] for indices in I]
    return similar_images



def combine_images(query_image, similar_images):
    images = [query_image] + [Image.open(img_path) for img_path in similar_images]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    combined_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return combined_image

index, image_paths = load_faiss_index(index_path, image_paths)

@app.post("/search/", response_class=FileResponse)
async def search(file: UploadFile = File(...), top_k: int = 5):
    image = Image.open(file.file).convert("RGB")
    similar_images = get_similar_images([image], index, image_paths, top_k=top_k, device='cpu')
    combined_image = combine_images(image, similar_images[0])

    # Save the combined image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        combined_image.save(tmp_file, format='PNG')
        tmp_file_path = tmp_file.name

    return FileResponse(tmp_file_path, media_type="image/png", filename="result.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
