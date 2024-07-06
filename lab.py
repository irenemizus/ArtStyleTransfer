import uuid

import neural_style_transfer
from quart import Quart, render_template, send_file, request, make_response
import os
import cv2
import numpy as np
from task_executor import Executor
from config import *

os.environ["QUART_APP"] = __file__
app = Quart(__name__)

app.jinja_env.globals.update(zip=zip)

executor = Executor(content_weight, style_weight, tv_weight, optimizer, model, init_method, iters_num, levels_num)

async def backend_task():
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')

    content_style_filename_pairs = [
        ('bird.jpg', 'painting_cut.jpg'),
        ('bird.jpg', 'expressive.jpg'),
        ('bird.jpg', 'starry_night.jpg'),
        ('bird.jpg', 'cubism2.jpg'),
        ('car.jpg', 'painting_cut.jpg'),
        ('car.jpg', 'expressive.jpg'),
        ('car.jpg', 'mona_lisa_prado.jpg'),
        ('car.jpg', 'cubism2.jpg'),
        ('columns.jpg', 'cubism1.jpg'),
        ('columns.jpg', 'cubism2.jpg'),
        ('columns.jpg', 'cubism3.jpg'),
        ('columns.jpg', 'painting_cut.jpg'),
        ('luda.jpg', 'mona_lisa.jpg'),
        ('luda.jpg', 'mosaic.jpg'),
        ('luda.jpg', 'starry_night.jpg'),
        ('luda.jpg', 'cubism1.jpg'),
        ('lion.jpg', 'mona_lisa.jpg'),
        ('lion.jpg', 'mosaic.jpg'),
        ('lion.jpg', 'starry_night.jpg'),
        ('lion.jpg', 'cubism1.jpg'),
    ]

    for pair in content_style_filename_pairs:
        content_img = neural_style_transfer.load_image(os.path.join(content_images_dir, pair[0]))
        style_img = neural_style_transfer.load_image(os.path.join(style_images_dir, pair[1]))

        await executor.add_task(str(uuid.uuid4()), neural_style_transfer.ContentStylePair((pair[0], content_img), (pair[1], style_img)))

    await executor.run()


@app.before_serving
async def startup():
    app.add_background_task(backend_task)



@app.route("/")
async def index():
    cards = []
    image_ids = await executor.task_ids()
    for image_id in image_ids:
        image_progress = await executor.get_progress(image_id)
        percent = image_progress[0] if image_progress[0] > 0 else 0
        cur_iter = percent / 100.0 * iters_num
        card = {
            "image_id": image_id,
            "percent": percent,
            "cur_iter": cur_iter,
            "iters_num": iters_num
        }

        cards.append(card)

    return await render_template('index.html', cards=cards)


@app.route('/generated/<image_id>', endpoint='generated')
async def serve_image(image_id):
    image_progress = await executor.get_progress(image_id)
    im = image_progress[1]
    if im is not None:
        # Encode the resized image to PNG
        im = np.clip(im * 255, 0, 255).astype('uint8')
        _, im_bytes_np = cv2.imencode('.png', im[:, :, ::-1])

        # Construct raw bytes string
        bytes_str = im_bytes_np.tobytes()

        # Create response given the bytes
        response = await make_response(bytes_str)
        response.headers['Content-Type'] = 'image/png'
    else:
        response = await make_response("No image yet")

    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
