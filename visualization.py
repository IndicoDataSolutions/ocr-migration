import cv2
import json


def visualize_results(file_to_mapping, new_ocr):
    for file_name, mapping in file_to_mapping.items():
        new_ocr_for_file = new_ocr[new_ocr["file_name"] == file_name]
        images = [
            cv2.imread(image_file)
            for image_file in json.loads(new_ocr_for_file["image_files"].values[0])
        ]

        for page_num, img in zip(mapping.keys(), images):
            for result in mapping[page_num]:
                t = result["token"]
                cv2.rectangle(
                    img,
                    (t["position"]["bbLeft"], t["position"]["bbBot"]),
                    (t["position"]["bbRight"], t["position"]["bbTop"]),
                    (255, 0, 0),
                    4,
                )
            cv2.imshow("Matched Labels", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
