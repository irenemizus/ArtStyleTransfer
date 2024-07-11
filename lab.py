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

# No-noise config, useless, just for demonstration
NO_NOISE_CONFIG = config.Config(
    noise_factor=0.0,
    noise_levels=                      (),
    noise_levels_central_amplitude=    (),
    noise_levels_peripheral_amplitude= (),
    noise_levels_dispersion =          ()
)

# Pixel-wide noise config, useless, just for demonstration
PIXEL_WIDE_NOISE_CONFIG = config.Config(
    noise_factor=0.5,
    noise_levels=                      (  -1, ),
    noise_levels_central_amplitude=    ( 1.0, ),
    noise_levels_peripheral_amplitude= ( 1.0, ),
    noise_levels_dispersion =          ( 0.5, )    # Doesn't matter
)

# 128-per-line noise config, useless, just for demonstration
NOISE_128_CONFIG = config.Config(
    noise_factor=0.7,
    noise_levels=                      ( 128, ),
    noise_levels_central_amplitude=    ( 1.0, ),
    noise_levels_peripheral_amplitude= ( 1.0, ),
    noise_levels_dispersion =          ( 0.5, )    # Doesn't matter
)

# 16-per-line noise config, useless, just for demonstration
NOISE_16_CONFIG = config.Config(
    noise_factor=0.7,
    noise_levels=                      (  16, ),
    noise_levels_central_amplitude=    ( 1.0, ),
    noise_levels_peripheral_amplitude= ( 1.0, ),
    noise_levels_dispersion =          ( 0.5, )    # Doesn't matter
)

# Gauss noise config, middle-sized image
# (the parameter values are specified in the declaration of the Config class)
STANDARD_GAUSS_NOISE_CONFIG = config.Config()

# Light gauss noise config, experimental one
LIGHT_GAUSS_NOISE_CONFIG = config.Config(
    content_weight=1e3,
    style_weight=1e3,
    tv_weight=0e0,
    levels_num=2,
    iters_num=1500,
    noise_factor=0.95,
    noise_levels=                      (  32,   64,  128,   -1,    0),
    noise_levels_central_amplitude=    (0.10, 0.15, 0.5, 0.10, 0.00),
    noise_levels_peripheral_amplitude= (0.20, 0.30, 0.10, 0.80, 0.00))

# The current lab config
config = STANDARD_GAUSS_NOISE_CONFIG

executor = Executor(config)

async def backend_task():
    """ A function for defining images to process """
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')

    content_style_filename_pairs = [
        ('bird.jpg', 'cubism2.jpg'),
        ('bird.jpg', 'matisse2.jpg'),
        ('bird.jpg', 'expressive.jpg'),
        ('bird.jpg', 'starry_night.jpg'),
        ('car.jpg', 'mosaic.jpg'),
        ('car.jpg', 'expressive.jpg'),
        ('car.jpg', 'matisse2.jpg'),
        ('car.jpg', 'cubism2.jpg'),
        ('columns.jpg', 'cubism1.jpg'),
        ('columns.jpg', 'cubism2.jpg'),
        ('columns.jpg', 'cubism3.jpg'),
        ('columns.jpg', 'matisse2.jpg'),
        ('girl_with_gun.jpg', 'mona_lisa.jpg'),
        ('girl_with_gun.jpg', 'mosaic.jpg'),
        ('girl_with_gun.jpg', 'starry_night.jpg'),
        ('girl_with_gun.jpg', 'cubism1.jpg'),
        ('lion.jpg', 'mona_lisa.jpg'),
        ('lion.jpg', 'mosaic.jpg'),
        ('lion.jpg', 'starry_night.jpg'),
        ('lion.jpg', 'cubism1.jpg'),
    ]


    for pair in content_style_filename_pairs:
        content_img = await load_image(os.path.join(content_images_dir, pair[0]))
        style_img = await load_image(os.path.join(style_images_dir, pair[1]))

        await executor.add_task(str(uuid.uuid4()), neural_style_transfer.ContentStylePair((pair[0], content_img), (pair[1], style_img)))


@app.before_serving
async def startup():
    app.add_background_task(backend_task)


async def load_image(img_path):
    """ A function for loading of an image """
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv2.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img

@app.route("/")
async def index():
    cards = []
    image_ids = await executor.task_ids()
    for image_id in image_ids:
        image_progress = await executor.get_progress(image_id)
        percent = image_progress[0] if image_progress[0] > 0 else 0
        cur_iter = percent / 100.0 * config.iters_num
        card = {
            "image_id": image_id,
            "percent": percent,
            "cur_iter": cur_iter,
            "iters_num": config.iters_num
        }

        cards.append(card)

    return await render_template('index.html', cards=cards)


@app.route('/generated/<image_id>', endpoint='generated')
async def serve_image(image_id):
    image_progress = await executor.get_progress(image_id)
    im = image_progress[1]
    if im is not None:
        # Encode the resized image to JPG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        im = np.clip(im * 255, 0, 255).astype('uint8')
        _, im_bytes_np = cv2.imencode('.jpg', im[:, :, ::-1], encode_param)

        # Construct raw bytes string
        bytes_str = im_bytes_np.tobytes()

        # Create response given the bytes
        response = await make_response(bytes_str)
        response.headers['Content-Type'] = 'image/jpg'
    else:
        response = await make_response("No image yet")

    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
