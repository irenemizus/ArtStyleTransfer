import asyncio
from typing import Callable
from processor import process_image
import uuid


class Task:
    def __init__(self, task_id: str, report: Callable, input_image_filename):
        self.__task_id = task_id
        self.__report = report

        self.__input_image_filename = input_image_filename

        self.job = asyncio.create_task(self.__do_job())

    async def __do_job(self):
        work_gen = process_image(self.__input_image_filename)
        async for result in work_gen:
            await asyncio.sleep(0.01)
            asyncio.get_running_loop().call_soon_threadsafe(self.__report, self.__task_id, result)


class Executor:
    def __init__(self):
        self.__tasks = {}
        self.__progress = {}
        self.__has_been_run = False

    def progress(self):
        return self.__progress

    def __print_progress(self):
        for task_id, p in self.__progress.items():
            print(str(p[0]) + ", ", end='')
        print()

    def __report(self, task_id, result):
        self.__progress[task_id] = result
        self.__print_progress()

    def add_task(self, task_id: str, input_image_filename: str):
        if self.__has_been_run:
            raise Exception('The backend is already running.')
        self.__progress[task_id] = (-1, None)
        self.__tasks[task_id] = Task(task_id=task_id, report=self.__report, input_image_filename=input_image_filename)

    async def run(self):
        self.__has_been_run = True
        jobs = [task.job for task in self.__tasks.values()]
        _ = await asyncio.wait(jobs)


async def main():
    tasks_count = 20
    executor = Executor()
    for i in range(tasks_count):
        executor.add_task(str(uuid.uuid4()), "custom_folder/luda1024.jpg")

    await executor.run()
    print('\nAll done.')


if __name__ == '__main__':
    # run the asyncio program
    asyncio.run(main())
