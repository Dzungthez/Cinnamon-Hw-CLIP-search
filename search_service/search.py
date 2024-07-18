from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import faiss
import torch
import clip
import numpy as np
from PIL import Image
import json
from typing import List
import matplotlib.pyplot as plt
import io
import base64
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
app = FastAPI()

def load_faiss_index(index_path, image_paths_path):
    index = faiss.read_index(index_path)
    with open(image_paths_path, "r") as f:
        image_paths = json.load(f)
    return index, image_paths

def get_similar_images(images, index, image_paths, top_k=5, model_name="ViT-B/32", device="cpu"):
    clip_model, clip_preprocess = clip.load(model_name, device=device)
    
    processed_images = torch.stack([clip_preprocess(img) for img in images]).to(device)
    
    with torch.no_grad():
        embeddings = clip_model.encode_image(processed_images)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy()
    
    D, I = index.search(embeddings, top_k)
    
    similar_images = [[image_paths[i] for i in indices] for indices in I]
    return similar_images

def display_images(query_images, similar_images):
    num_queries = len(query_images)
    num_similar = len(similar_images[0])
    
    if num_queries == 1:
        fig, axs = plt.subplots(1, num_similar + 1, figsize=(10, 5))
        axs = [axs]  # Make it iterable
    else:
        fig, axs = plt.subplots(num_queries, num_similar + 1, figsize=(15, 5 * num_queries))
    
    for i, query_image in enumerate(query_images):
        axs[i][0].imshow(query_image)
        axs[i][0].set_title("Query Image")
        axs[i][0].axis("off")
        
        for j, sim_image in enumerate(similar_images[i]):
            axs[i][j + 1].imshow(Image.open(sim_image))
            axs[i][j + 1].set_title(f"Similar Image {j + 1}")
            axs[i][j + 1].axis("off")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

index, image_paths = load_faiss_index("../data/index.faiss", "../data/image_paths.json")

@app.post("/search/", response_class=HTMLResponse)
async def search(file: UploadFile = File(...), top_k: int = 5):
    image = Image.open(file.file).convert("RGB")
    similar_images = get_similar_images([image], index, image_paths, top_k=top_k, device = 'cuda')
    img_str = display_images([image], similar_images)
    html_content = f'<img src="data:image/png;base64,{img_str}" alt="Result Image"/>'
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
