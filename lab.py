import uuid

from task_executor import Executor
from quart import Quart, render_template, send_file, request, make_response
import os
import cv2

os.environ["QUART_APP"] = __file__
app = Quart(__name__)

executor = Executor()


async def backend_task():
    tasks_count = 8
    for i in range(tasks_count):
        executor.add_task(str(uuid.uuid4()), "custom_folder/luda1024.jpg")

    await executor.run()


@app.before_serving
async def startup():
    app.add_background_task(backend_task)
    pass


@app.route("/")
async def index():
    image_ids = executor.progress().keys()
    return await render_template('index.html', image_ids=image_ids)


@app.route('/generated/<image_id>', endpoint='generated')
async def serve_image(image_id):
    progress = executor.progress()
    image_progress = progress[image_id]
    im = image_progress[1]
    if im is not None:

        # Encode the resized image to PNG
        _, im_bytes_np = cv2.imencode('.png', im)

        # Constuct raw bytes string
        bytes_str = im_bytes_np.tobytes()

        # Create response given the bytes
        response = await make_response(bytes_str)
        response.headers['Content-Type'] = 'image/png'
    else:
        response = await make_response("No image yet")

    return response


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
