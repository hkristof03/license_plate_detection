import os
import string

import pandas as pd
import cv2
import pytesseract

import utils


def detect_license_plates(plot: bool = False):
    """

    :return:
    """
    multiple_detections = []
    # license_plate_characters
    lpc = list(string.ascii_uppercase) + list(str(i) for i in range(0, 10))
    df_labels = utils.read_df_labels()
    image_paths = utils.list_images()

    for img_path in image_paths:

        img_id = img_path.split(os.sep)[-1]
        condition = (df_labels['img_id'] == img_id)
        desired_label = df_labels.loc[condition]['license_plate'].values[0]

        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur the background
        blur = cv2.bilateralFilter(gray, 11, 90, 90)
        edges = cv2.Canny(blur, 30, 100)
        cnts, hierarchy = cv2.findContours(edges, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
        #cnts_filtered = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        cnts_filtered = sorted(cnts, key=cv2.contourArea, reverse=True)

        plates = []
        plates_gray = []

        for c in cnts_filtered:

            perimeter = cv2.arcLength(c, True)
            edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(edges_count) == 4:
                x, y, w, h = cv2.boundingRect(c)

                if h / w < 0.25:
                    plate = image[y: y + h, x: x + w]
                    plate_gray = gray[y: y + h, x: x + w]

                    plates.append(plate)
                    plates_gray.append(plate_gray)

        actual_label = ''

        if plates_gray:
            for i, plate in enumerate(plates_gray):
                # Normalize and threshold image
                plate = cv2.normalize(plate, None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX)
                res, plate = cv2.threshold(plate, 100, 255, cv2.THRESH_BINARY)

                text = pytesseract.image_to_string(plate, lang="eng",
                                                   config='--psm 7')
                text = ''.join(list(filter(lambda x_: x_ in lpc, text)))

                if len(text) > 4 and text != actual_label:
                    actual_label = text

                    if actual_label == desired_label:
                        df_labels.loc[
                            condition, 'detected_labels'
                        ] = actual_label
                    else:
                        multiple_detections.append(
                            {
                                'img_id': img_id,
                                'detected_label': actual_label
                            }
                        )
                    print(f"Detected text for img: {img_id} is: {text}")

                    if plot:
                        #utils.plot_images(plate, plates[i])
                        image_copy = image.copy()
                        _ = cv2.drawContours(image_copy, cnts_filtered, -1,
                                             (0, 255, 0), 2)
                        utils.plot_images2(image, gray, blur, edges,
                                           image_copy, plate, text)
        else:
            print(f"There was no successful detection for image: {img_id}")

    df_labels.to_csv(os.path.join(os.getcwd(), 'detected_labels1.csv'),
                     index=False)
    df_wrong_detections = pd.DataFrame(multiple_detections)
    df_wrong_detections.to_csv(
        os.path.join(os.getcwd(), 'wrong_detections1.csv'), index=False
    )


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = os.path.join(
        'C:\\', 'Program Files', 'Tesseract-OCR', 'tesseract.exe'
    )

    detect_license_plates(plot=False)
