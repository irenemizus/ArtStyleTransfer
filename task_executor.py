import asyncio
import time
from typing import Callable

from neural_style_transfer import neural_style_transfer, ContentStylePair

from config import simultaneous_tasks_count

sem = asyncio.Semaphore(simultaneous_tasks_count)



class Task:
    """ A class representing a single optimization task and reporting its output to the Executor """

    def __init__(self, content_n_style: ContentStylePair, config,
                 task_id: str, report: Callable, job_done: Callable):
        self.__task_id = task_id
        self.__report = report
        self.__job_done_callback = job_done

        self.__content_n_style = content_n_style
        self.__config = config

        self.job = asyncio.create_task(self.__do_job())


    async def __do_job(self):
        print(f'Processing content image {self.__content_n_style.content[0]}, style image {self.__content_n_style.style[0]}; initial method: {self.__config.init_method}')
        async with sem:
            print("awaiting neural_style_transfer")
            async for result in neural_style_transfer(self.__content_n_style,
                                                       self.__config.content_weight, self.__config.style_weight, self.__config.tv_weight,
                                                       self.__config.optimizer, self.__config.model, self.__config.init_method,
                                                       self.__config.iters_num, self.__config.levels_num, self.__config.noise_factor,
                                                       self.__config.noise_levels, self.__config.noise_levels_central_amplitude,
                                                       self.__config.noise_levels_peripheral_amplitude,
                                                       self.__config.noise_levels_dispersion):
                result_copy = (result[0], result[1].copy())
                await self.__report(self.__task_id, result_copy)

            await self.__job_done_callback(self.__task_id)


class Executor:
    """ The class executing the optimization tasks. And collecting the results  """

    def __init__(self, config, report_progress=None):
        self.__tasks = {}
        self.__progress = {}

        self.__config = config

        self.__progress_lock = asyncio.Lock()
        self.__tasks_lock = asyncio.Lock()

        self.__report_progress = report_progress


    async def get_progress(self, key):
        async with self.__progress_lock:
            value = self.__progress[key]
            # (progress(%), current optimizing image)
            vv = (value[0], value[1].copy() if value[1] is not None else None)
            return vv


    async def progress(self):
        async with self.__progress_lock:
            for pr in self.__progress.items():
                yield pr


    async def task_ids(self):
        res = []
        async with self.__progress_lock:
            for k in self.__progress.keys():
                res.append(k)
        return res


    async def set_progress(self, key, value):
        async with self.__progress_lock:
            # (progress(%), current optimizing image)
            vv = (value[0], value[1].copy() if value[1] is not None else None)
            self.__progress[key] = vv


    async def __print_progress(self):
        async for task_id, p in self.progress():
            print("Progress: " + str(task_id) + ", " + str(p[0]))
        print()


    async def __report(self, task_id, result):
        await self.set_progress(task_id, result)
        await self.__print_progress()
        if self.__report_progress is not None:
            await self.__report_progress(task_id, result)


    async def __job_done(self, task_id):
        async with self.__tasks_lock:
            print(f"Task {task_id} done")
            self.__tasks.pop(task_id)


    async def add_task(self, task_id: str, content_n_style: ContentStylePair):
        await self.set_progress(task_id, (-1, None))
        async with self.__tasks_lock:
            self.__tasks[task_id] = Task(content_n_style,
                                         self.__config, task_id=task_id, report=self.__report, job_done=self.__job_done)
            print(f"Task {task_id} run")
            return self.__tasks[task_id].job

    async def run(self, forever=False):
        """ A running function for Telegram bot """
        while forever:
            waiting_print = False
            while True:
                async with self.__tasks_lock:
                    any_tasks_left = len(self.__tasks) > 0
                    if not any_tasks_left: break
                    jobs = [task.job for task in self.__tasks.values()]
                waiting_print = True
                await asyncio.wait(jobs)
            if waiting_print:
                print("No more tasks in the queue. Waiting for the new ones...")
            time.sleep(1)

