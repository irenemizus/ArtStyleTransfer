import asyncio
import os
import uuid

import neural_style_transfer
from lab import load_image

from task_executor import Executor
import config

STARTING_CONFIG = config.Config(
    levels_num=1,
    iters_num=10
)

# The current lab config
config = STARTING_CONFIG

executor = Executor(config)

async def main():
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')

    content_style_filename_pairs = [
        ('bird.jpg', 'cubism2.jpg'),
    ]

    jobs = []
    for pair in content_style_filename_pairs:
        content_img = await load_image(os.path.join(content_images_dir, pair[0]))
        style_img = await load_image(os.path.join(style_images_dir, pair[1]))

        jobs.append(await executor.add_task(str(uuid.uuid4()), neural_style_transfer.ContentStylePair((pair[0], content_img), (pair[1], style_img))))

    await executor.run()
    print("All jobs done")

if __name__ == '__main__':
    asyncio.run(main=main())