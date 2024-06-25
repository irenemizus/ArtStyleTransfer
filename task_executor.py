import asyncio
from typing import Callable
#from processor import process_image
import uuid

from neural_style_transfer import neural_style_transfer

sem = asyncio.Semaphore(2)


class Task:
    def __init__(self, content_img_filename, style_img_filename, height, content_weight, style_weight, tv_weight, optimizer, model, init_method, content_images_dir, style_images_dir, output_img_dir, img_format, task_id: str, report: Callable):
        self.__task_id = task_id
        self.__report = report

        self.__content_img_filename = content_img_filename
        self.__style_img_filename = style_img_filename
        self.__height = height
        self.__content_weight = content_weight
        self.__style_weight = style_weight
        self.__tv_weight = tv_weight
        self.__optimizer = optimizer
        self.__model = model
        self.__init_method = init_method
        self.__content_images_dir = content_images_dir
        self.__style_images_dir = style_images_dir
        self.__output_img_dir = output_img_dir
        self.__img_format = img_format

        self.job = asyncio.create_task(self.__do_job())

    async def __do_job(self):

        print(f'Processing {self.__content_img_filename}')
        async with sem:
            print("in sem")
            #for content_img_filename, style_img_filename in zip(self.__content_img_filenames, self.__style_img_filenames):
            print("awaiting neural_style_transfer")
            async for result in neural_style_transfer(self.__content_img_filename, self.__style_img_filename,
                                                           self.__height,
                                                           self.__content_weight, self.__style_weight, self.__tv_weight,
                                                           self.__optimizer, self.__model, self.__init_method,
                                                           self.__content_images_dir, self.__style_images_dir,
                                                           self.__output_img_dir, self.__img_format):
                result_copy = (result[0], result[1].copy())
                await self.__report(self.__task_id, result_copy)



class Executor:
    def __init__(self, height, content_weight, style_weight, tv_weight, optimizer, model, init_method, content_images_dir, style_images_dir, output_img_dir, img_format):
        self.__tasks = {}
        self.__progress = {}
        self.__has_been_run = False

        self.__height = height
        self.__content_weight = content_weight
        self.__style_weight = style_weight
        self.__tv_weight = tv_weight
        self.__optimizer = optimizer
        self.__model = model
        self.__init_method = init_method
        self.__content_images_dir = content_images_dir
        self.__style_images_dir = style_images_dir
        self.__output_img_dir = output_img_dir
        self.__img_format = img_format

        self.__lock = asyncio.Lock()

    async def get_progress(self, key):
        async with self.__lock:
            value = self.__progress[key]
            vv = (value[0], value[1].copy() if value[1] is not None else None)
            return vv

    async def progress(self):
        async with self.__lock:
            for pr in self.__progress.items():
                yield pr

    async def task_ids(self):
        res = []
        async with self.__lock:
            for k in self.__progress.keys():
                res.append(k)
        return res

    async def set_progress(self, key, value):
        async with self.__lock:
            vv = (value[0], value[1].copy() if value[1] is not None else None)
            self.__progress[key] = vv


    async def __print_progress(self):
        async for task_id, p in self.progress():
            #q = p[1].copy()

            #a = p[1][0][0][0]
            #print(type(p[1]))
            print("Progress: " + str(task_id) + ", " + str(p[0])) # + ", " + str(q[0][0][0]))
            #print("something")
            pass
        print()

    async def __report(self, task_id, result):
        await self.set_progress(task_id, result)
        await self.__print_progress()

    async def add_task(self, task_id: str, content_img_filename, style_img_filename):
        if self.__has_been_run:
            raise Exception('The backend is already running.')
        await self.set_progress(task_id, (-1, None))
        self.__tasks[task_id] = Task(content_img_filename, style_img_filename, self.__height,
                                     self.__content_weight, self.__style_weight, self.__tv_weight,
                                     self.__optimizer, self.__model, self.__init_method,
                                     self.__content_images_dir, self.__style_images_dir, self.__output_img_dir, self.__img_format,
                                     task_id=task_id, report=self.__report)

    async def run(self):
        self.__has_been_run = True
        jobs = [task.job for task in self.__tasks.values()]
        _ = await asyncio.wait(jobs)


async def main():
    pass
#     tasks_count = 20
#     executor = Executor()
#     for i in range(tasks_count):
#         executor.add_task(str(uuid.uuid4()), "custom_folder/luda1024.jpg")
#
#     await executor.run()
#     print('\nAll done.')


if __name__ == '__main__':
    # run the asyncio program
    asyncio.run(main())
