import asyncio
import copy
import logging
import sys
import traceback
import uuid
from asyncio import Lock
from typing import List

import cv2
import numpy as np


from aiogram import Bot, Dispatcher, html, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, InputFile, CallbackQuery, BufferedInputFile
from aiogram.methods import SendPhoto

from aiogram_media_group import media_group_handler, MediaGroupFilter

import config
import neural_style_transfer
from task_executor import Executor

from token_DO_NOT_COMMIT import TOKEN
# The file token_DO_NOT_COMMIT.py should look like this:
# TOKEN = "7433346137:AAF2vjCKBNK_WlXJKBR1_7qFIWN4G5KExyE"
# The bot token should be obtained via https://t.me/BotFather


# Initialize Bot instance with default bot properties which will be passed to all API calls
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()

class ChatProgress:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.progress = 0

tasks_table: dict[str, ChatProgress] = {}
table_lock: Lock = Lock()


class NotTwoPhotosException(Exception):
    pass


async def task_progress_callback(task_id, result):
    try:
        percent = result[0]
        res_img = result[1]

        new_img_np = res_img[:, :, ::-1]  # RGB -> BGR

        new_img_np = np.clip(new_img_np * 255, 0, 255).astype('uint8')
        _, new_res = cv2.imencode('.jpg', new_img_np)
        new_res_bytes = new_res.tobytes()

        async with table_lock:
            chat_id = tasks_table[task_id].chat_id
            old_percent = tasks_table[task_id].progress

        if percent - old_percent >= 20 or percent >= 100:
            image_file = BufferedInputFile(new_res_bytes, f'image_{percent:.1f}.jpg')
            caption = f"Progress: {percent:.1f}%"
            if percent >= 100:
                caption = "Done!"
            await bot(SendPhoto(chat_id=chat_id, photo=image_file, caption=caption))
            async with table_lock:
                tasks_table[task_id].progress = percent

        async with table_lock:
            if percent >= 100:
                del tasks_table[task_id]

    except:
        traceback.print_exc()
        raise

config = config.Config()
executor = Executor(config.content_weight, config.style_weight, config.tv_weight, config.optimizer, config.model, config.init_method, config.iters_num, config.levels_num, config.noise_factor, report_progress=task_progress_callback)


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}! "
                         f"To start a job please send me two pictures {html.italic('in a single message')} - one for the {html.bold('content')} and one for the {html.bold('style')}")


async def respond_with_send_me_two_pictures(message: types.Message):
    await message.answer(
        f"To start a job please send me two pictures {html.italic('in a single message')} - one for the {html.bold('content')} and one for the {html.bold('style')}")


@dp.message(MediaGroupFilter())  #MediaGroupFilter(is_media_group=True), content_types=ContentType.PHOTO)
@media_group_handler
async def album_handler(messages: List[types.Message]):
    try:
        if len(messages) != 2:
            raise NotTwoPhotosException

        images_to_process = []

        for message in messages:
            if message.photo:
                file = await bot.get_file(message.photo[-1].file_id)
                result = await bot.download_file(file.file_path)

                with result as f:
                    res_np = np.frombuffer(f.read(), np.uint8)
                    img_np = cv2.imdecode(res_np, cv2.IMREAD_COLOR)
                img_np = img_np[:, :, ::-1]  # BGR -> RGB

                img_np = img_np.astype(np.float32)  # convert from uint8 to float32
                img_np /= 255.0  # get to [0, 1] range

                images_to_process.append(img_np)

        if len(images_to_process) != 2:
            raise NotTwoPhotosException

        content_img = copy.deepcopy(images_to_process[0])
        style_img = copy.deepcopy(images_to_process[1])

        task_id = str(uuid.uuid4())

        async with table_lock:
            assert messages[0].chat.id == messages[1].chat.id, "Messages are from different chats? How?"
            tasks_table[task_id] = ChatProgress(messages[-1].chat.id)
        await messages[-1].answer("Processing has started. Please, wait...")

        await executor.add_task(task_id,
                                neural_style_transfer.ContentStylePair(('content.jpg', content_img),
                                                                       ('style.jpg', style_img)))

    except NotTwoPhotosException:
        await respond_with_send_me_two_pictures(messages[-1])

    except:
        traceback.print_exc()
        await messages[-1].answer("Oops... Something went wrong on the server. Please ask the developer to check the logs")


@dp.message()
async def not_album_handler(message: types.Message):
    await respond_with_send_me_two_pictures(message)


async def backend_task():
    return executor.run(forever=True)


async def on_bot_start_up(dispatcher: Dispatcher) -> None:
    """List of actions which should be done before bot start"""
    await asyncio.create_task(backend_task())  # creates background task


async def main() -> None:
    # And the run events dispatching
    await dp.start_polling(bot, on_startup=on_bot_start_up)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
