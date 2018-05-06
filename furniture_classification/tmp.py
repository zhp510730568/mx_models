import os
from mxnet import image

count = 0
image_path='./train_dir/'
for label in os.listdir(image_path):
    label_path = os.path.join(image_path, '56')
    for image_name in os.listdir(label_path):
        image_file = os.path.join(label_path, image_name)
        print(image_file)
        try:
            img = image.imdecode(open(image_file, 'rb').read()).astype('float32')
            count += 1
            print(count)
        except Exception as ex:
            print("error: ", ex)