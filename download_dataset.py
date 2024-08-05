from huggingface_hub import hf_hub_download
import zipfile

data_path = 'data'

download_path = hf_hub_download(repo_id='huudung141434/coco128', 
                                    filename='COCO-128-2.zip', repo_type='space',
                                   cache_dir=data_path)
print(f"File downloaded to {download_path}")
    
with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
print(f"Files extracted to {data_path}")