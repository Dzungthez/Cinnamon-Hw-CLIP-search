from locust import HttpUser, task, between
from fastapi import UploadFile, File

class MyUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def search_test(self):
        top_k = 5
        file: UploadFile = File(...)
        args = {
            "file": file,
            "top_k": top_k
        }
        self.client.post("/search/", json=args)

    @task
    def encode_test(self):
        add_data = False
        file: UploadFile = File(...)
        args = {
            "file": file,
            "add_data": add_data
        }
        self.client.post("/encode/", json=args)