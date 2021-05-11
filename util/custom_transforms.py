import cv2
import numpy as np


class UserSimulator(object):
    def __init__(self, clusters=10, maxLinesNumber=50, imsize=256):
        self.clusters = clusters
        self.maxLinesNumber = maxLinesNumber
        self.angleScale = np.pi / 20
        self.seed_number = 0
        self.imsize = imsize

        self.numsamples = 100_000
        self.imshape = np.array([self.imsize, self.imsize])

        self.start_coords = np.random.normal(
            loc=self.imshape / 2,
            scale=self.imshape / 4,
            size=(self.numsamples)
        )
        self.start_angles = np.random.uniform(2 * np.pi, size=self.numsamples)
        self.numVs = np.random.randint(self.maxLinesNumber, size=self.numsamples)
        self.lengths = np.random.uniform(self.maxLength, size=self.numsamples)
        self.norm_deltas = np.random.normal(size=self.numsamples)
        self.angle_deltas = np.random.normal(loc=0.0, scale=self.angleScale, size=self.numsamples)

    def __call__(self):
        mask = np.full((self.imsize, self.imsize, 1), 0.0)
        self.maxLength = self.imsize / 4

        for i in range(self.clusters):
            start_coord = self.start_coords[self.seed]
            end_coord = np.zeros_like(start_coord)
            startAngle = self.start_angles[self.seed]

            numV = self.numVs[self.seed]
            length = self.lengths[self.seed]
            for j in range(numV):
                length += self.norm_deltas[self.seed] * length / 10

                angle = startAngle + self.angle_deltas[self.seed]

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
        print("usersimulator called", start_coord, end_coord, length)
        return (mask == 0).astype(np.float32)
    
    @property
    def seed(self):
        self.seed_number += 1
        return self.seed_number

