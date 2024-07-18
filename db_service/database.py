from fastapi import FastAPI, UploadFile, File
import faiss
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import clip
import json
from torch.utils.data import DataLoader, Dataset
from typing import List
import glob
from typing import Union, List
app = FastAPI()

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.preprocess(image)
        return image

def get_data_paths(dir: Union[str, List[str]], data_formats: List[str], prefix: str = '') -> List[str]:
    try:
        f = []  # data files
        for d in dir if isinstance(dir, list) else [dir]:
            p = Path(d)
            if p.is_dir():
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            else:
                raise FileNotFoundError(f'{prefix}{p} does not exist')
        data_files = sorted(x for x in f if x.split('.')[-1].lower() in data_formats)
        return data_files
    except Exception as e:
        raise Exception(f'{prefix}Error loading data from {dir}: {e}') from e

@app.post("/build-index/")
async def build_index(data_dir: str, model_name: str = "ViT-B/32", batch_size: int = 4, device: str = "cpu"):
    model, preprocess = clip.load(model_name, device=device)
    
    image_paths = get_data_paths(data_dir, data_formats=["jpg", "jpeg", "png"])
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    image_embeddings = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            image_embeddings.append(embeddings.cpu().numpy())

    image_embeddings = np.vstack(image_embeddings)
    
    d = image_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(image_embeddings)
    
    faiss.write_index(index, "/data/index.faiss")
    with open("data/image_paths.json", "w") as f:
        json.dump(image_paths, f, indent=4)

    return {"status": "index built successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
