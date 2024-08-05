from locust import HttpUser, TaskSet, task, between
import os

class UserBehavior(TaskSet):
    @task(1)
    def search(self):
        # Đường dẫn đến file bạn muốn gửi
        file_path = "result (1).png"
        
        # Mở file dưới dạng binary
        with open(file_path, "rb") as f:
            # Gửi file dưới dạng form data
            files = {'file': (os.path.basename(file_path), f, 'image/png')}
            self.client.post("/search/", files=files)

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --host=http://35.155.57.220:8001")
