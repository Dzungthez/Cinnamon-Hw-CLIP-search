#### Cinnamon-Hw-CLIP-search

This is a small project for image search. Users can upload images and find the top-k most similar images stored in the database, which contains about 300 images from the COCO-128 dataset. There is also an option for users to add their own images to the database.

##### Accessing the services (Locally)

To build with Docker and run the service, follow these steps:

bash
Copy code
docker compose up
Then, open your browser and go to: localhost:8001/search. Here, you can upload your images and start searching for similar ones.

Also, visit localhost:8001/encode. You can upload images, see the embedding results, and have the option to save your images to the vector database.

##### Overview of the Architecture

Dataset Download: The project runs encode_service/download_dataset.py, which automatically downloads the COCO-128 dataset and places it in the data folder.
Database Creation: The database.py file is used to create the index.faiss file, which is the index for our database.
We provide two FastAPI services:

Encoding Service (encode_service/encode.py): This handles encoding of images.
Searching Service (search_service/search.py): This handles searching for similar images.
Both services use the CLIP model and include simple processing logic for loading and displaying images to meet user needs.

##### Demo

![alt text](<result (1).png>)
