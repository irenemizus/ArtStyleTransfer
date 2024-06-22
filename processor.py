import asyncio
import concurrent

import numpy as np
import cv2


sem = asyncio.Semaphore(4)


async def process_image(input_image_filename):
    async with sem:
        im = cv2.imread(input_image_filename, cv2.IMREAD_ANYCOLOR).astype(np.float32)

        for i in range(100):
            im *= 0.99
            im += 0.01 * np.random.normal(0.5,1.0, im.shape)

            result = (i, im)
            yield result