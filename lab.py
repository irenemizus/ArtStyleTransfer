import uuid

from task_executor import Executor
from quart import Quart, render_template, send_file, request, make_response
import os
import cv2
import numpy as np

os.environ["QUART_APP"] = __file__
app = Quart(__name__)

app.jinja_env.globals.update(zip=zip)

default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
content_images_dir = os.path.join(default_resource_dir, 'content-images')
style_images_dir = os.path.join(default_resource_dir, 'style-images')

content_img_filenames = ['luda.jpg', 'lion.jpg']
style_img_filenames = ['mona_lisa.jpg', 'mona_lisa.jpg', 'vg_starry_night.jpg', 'ben_giles.jpg']
height = 256
content_weight = 1e1
style_weight = 1e5
tv_weight = 0e3
optimizer = 'lbfgs'
model = 'vgg19'
init_method = 'content'
tasks_count = 2
levels_num = 2
iters_num = 20

executor = Executor(height, content_weight, style_weight, tv_weight, optimizer, model, init_method, content_images_dir, style_images_dir, iters_num, levels_num)


async def backend_task():
    for i in range(tasks_count):
        await executor.add_task(str(uuid.uuid4()),
                                content_img_filename=content_img_filenames[i],
                                style_img_filename=style_img_filenames[i])

    await executor.run()


@app.before_serving
async def startup():
    app.add_background_task(backend_task)


@app.route("/")
async def index():
    image_prog = []
    prog_data = []
    image_ids = await executor.task_ids()
    for image_id in image_ids:
        image_progress = await executor.get_progress(image_id)
        prog_data.append(image_progress[0] if image_progress[0] > 0 else 0)
        cur_iter = prog_data[0] / 100.0 * iters_num
        prog_data.extend([cur_iter, iters_num])
        image_prog.append(prog_data)
    return await render_template('index.html', image_ids=image_ids, image_prog=image_prog)


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
    app.run(host="127.0.0.1", port=8080, debug=True)
