import uuid

import neural_style_transfer
from quart import Quart, render_template, make_response
import os
import cv2
import numpy as np
from task_executor import Executor
import config

os.environ["QUART_APP"] = __file__
app = Quart(__name__)

app.jinja_env.globals.update(zip=zip)

STARTING_CONFIG = config.Config(
    levels_num=1,
    iters_num=10
)

# The current lab config
config = STARTING_CONFIG

executor = Executor(config.content_weight, config.style_weight, config.tv_weight,
                    config.optimizer, config.model, config.init_method,
                    config.iters_num, config.levels_num, config.noise_factor,
                    config.noise_levels, config.noise_levels_central_amplitude,
                    config.noise_levels_peripheral_amplitude, config.noise_levels_dispersion)

async def backend_task():
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')

    content_style_filename_pairs = [
        ('bird.jpg', 'cubism2.jpg'),
    ]

    for pair in content_style_filename_pairs:
        content_img = await load_image(os.path.join(content_images_dir, pair[0]))
        style_img = await load_image(os.path.join(style_images_dir, pair[1]))

        await executor.add_task(str(uuid.uuid4()), neural_style_transfer.ContentStylePair((pair[0], content_img), (pair[1], style_img)))

    #await executor.run()


@app.before_serving
async def startup():
    app.add_background_task(backend_task)


async def load_image(img_path):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv2.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
