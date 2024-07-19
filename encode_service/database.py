import faiss
import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
import clip
import json
from torch.utils.data import DataLoader, Dataset
from typing import List
import glob
from typing import Union, List
import dotenv
dotenv.load_dotenv()

model_name = os.getenv("MODEL_NAME")
data_path = os.getenv("DATA_PATH")
index_path = os.getenv("INDEX_PATH")
image_paths = os.getenv("IMAGES_PATH")

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
        f = [] 
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

def build_index(data_dir: str, model_name: str = model_name, batch_size: int = 4, device: str = "cpu"):
    model, preprocess = clip.load(model_name, device=device)
    
    img_path_list = get_data_paths(data_dir, data_formats=["jpg", "jpeg", "png"])
    dataset = ImageDataset(img_path_list, preprocess)
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
    
    faiss.write_index(index, index_path)
    with open(image_paths, "w") as f:
        json.dump(img_path_list, f, indent=4)

    return {"status": "index built successfully"}

if __name__ == "__main__":
    print(build_index(data_path, model_name))
    # print(model_name, data_path, index_path, image_paths, sep='\n')
