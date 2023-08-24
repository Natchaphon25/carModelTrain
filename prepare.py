import cv2
import requests
import numpy as np
import pickle
import os
import base64


url = "http://localhost:8080/api/genhog/"

def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "data:image/jpeg;base64,"+str.split(str(img_str),"'")[1]
    # print("img_str: ",data)
    response = requests.post(url, json={"image_str": data})

    if response.status_code == 200:
        try:
            result_json = response.json()
            return result_json
        except ValueError:
            print("Error: Unable to decode JSON from response.")
    else:
        print("Error:", response.status_code, response.text)
    return None
   
    # return response.json()
# path = "train\Audi\1.jpg"

# img = cv2.imread('train\\Audi\\1.jpg')
# print(img2vec(img))


path = 'train'
#ข้อมูลภาพ
x = []
#label ยี่ห้อรถ
y = []
featurevector = []

for brand in os.listdir(path):
  for car in os.listdir(os.path.join(path,brand)):
    img_file_name = os.path.join(path,brand)+"/"+car
    img = cv2.imread(img_file_name)
    x.append(img)
    y.append(brand)
#     res = img2vec(img)
#     vec = list(res["HOG Descriptor"])
#     vec.append(brand)
#     featurevector.append(vec)
print(x)
print(y)

# write_path = "trainfeaturevector.pkl"
# pickle.dump(featurevector, open(write_path,"wb"))
# print("data preparation is done")
