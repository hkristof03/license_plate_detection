import os
import glob
import pandas as pd
import cv2


ROOT_DIR = os.path.join(os.getcwd(), 'baza_slika')
directories = os.listdir(ROOT_DIR)
file_labels = os.path.join(os.getcwd(), 'img_labels.csv')
df = pd.DataFrame()
images = []

for directory in directories:
    path_images = os.path.join(ROOT_DIR, directory)
    if os.path.isdir(path_images):
        imgs_ = glob.glob(os.path.join(path_images, '*.jpg'))
        images.extend(imgs_)

n_imgs = len(images)

if os.path.isfile(file_labels) and df.empty:
    df = pd.read_csv(file_labels)
    stored_images = dict(zip(df['img_id'], df['license_plate']))
    images = [
        img for img in images if img.split('\\')[-1] not in stored_images
    ]

print(f"{n_imgs - len(images)} images are already labeled!")
n_imgs = len(images)

for img_path in images:

    img_id = img_path.split('\\')[-1]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[0] > 600 or img.shape[1] > 800:
        img = cv2.resize(img, None, fx=0.75, fy=0.75)

    cv2.imshow(f"{img_id}", img)
    cv2.waitKey(2000)
    value = input(f"{n_imgs} images are left, enter license plate number: ")
    cv2.destroyAllWindows()
    d = {
        'img_id': img_id,
        'license_plate': value
    }
    df_ = pd.DataFrame([d])
    df = pd.concat([df, df_], axis=0)
    df.to_csv(file_labels, index = False)
    n_imgs -= 1