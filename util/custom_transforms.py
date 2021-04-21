import torch
import cv2
import numpy as np
from torchvision import transforms


class UserSimulator(object):
    def __init__(self, clusters=10, maxLinesNumber=50):
        self.clusters = clusters
        self.maxLinesNumber = maxLinesNumber
        self.angleScale = np.pi / 20

    def __call__(self, img):
        mask = np.full((img.shape[0], img.shape[1], 1), 0.0)
        self.maxLength = mask.shape[0] / 4

        for i in range(self.clusters):
            start_coord = np.random.normal(
                loc=np.array(mask.shape[:2]) / 2, scale=np.array(mask.shape[:2]) / 4
            )
            end_coord = np.zeros_like(start_coord)
            startAngle = np.random.uniform(2 * np.pi)

            numV = np.random.randint(self.maxLinesNumber)
            length = np.random.uniform(self.maxLength)
            for j in range(numV):
                length += np.random.normal() * length / 10

                angle = startAngle + np.random.normal(loc=0.0, scale=self.angleScale)

                if j % 2:
                    angle += np.pi

                end_coord[0] = start_coord[0] + length * np.sin(angle)
                end_coord[1] = start_coord[1] + length * np.cos(angle)

                cv2.line(
                    mask,
                    tuple(start_coord.astype(int)),
                    tuple(end_coord.astype(int)),
                    (255, 255, 255),
                    thickness=mask.shape[0] // 40,
                )
                start_coord = end_coord.copy()
        return (mask > 0).astype(np.float32)


if __name__ == "__main__":
    img = cv2.imread("./CelebAMask-HQ/imgs/120.jpg")
    tr = transforms.Compose([UserSimulator()])
    cv2.imwrite("./tmp.png", tr(img))
