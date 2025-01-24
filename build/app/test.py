import base64
import requests

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def post_image_base64(image_path):
    url = "http://0.0.0.0:6969/process"
    image_b64 = image_to_base64(image_path)
    payload = {"image": image_b64}

    response = requests.post(url, json=payload)
    print("Response Status Code:", response.status_code)
    print("Response Body:", response.json())

if __name__ == "__main__":
    post_image_base64("cropped_image.png")
