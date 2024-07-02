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
from aiogram.enums import ParseMode, ContentType
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, InputFile, CallbackQuery, BufferedInputFile
from aiogram.methods import SendPhoto
from magic_filter import MagicFilter
from typing_extensions import Union

import neural_style_transfer
from task_executor import Executor
from config import *

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

executor = Executor(content_weight, style_weight, tv_weight, optimizer, model, init_method, iters_num, levels_num, report_progress=task_progress_callback)


# This class contains data for an incomplete task
class TaskCollector:
#    STYLE = "style"
#    CONTENT = "content"

    def __init__(self, media_group_id: str):
#        self.value = BotState.CONTENT
        self.__cont_img = None
        self.__style_img = None
        self.media_group_id = media_group_id

        self.__imgs_dict = {}

    def add_image(self, message_id, image):
        if message_id not in self.__imgs_dict:
            self.__imgs_dict[message_id] = image

        if len(self.__imgs_dict) > 2:
            #self.__imgs_dict = {}
            raise Exception

        if len(self.__imgs_dict) == 2:
            ids = list(self.__imgs_dict.keys())
            delta_ids = ids[1] - ids[0]
            if delta_ids == 1:
                self.__cont_img = self.__imgs_dict[ids[0]]
                self.__style_img = self.__imgs_dict[ids[1]]
            elif delta_ids == -1:
                self.__cont_img = self.__imgs_dict[ids[1]]
                self.__style_img = self.__imgs_dict[ids[0]]
            else:
                #self.__imgs_dict = {}
                raise Exception


    # def set_cont_img(self, cont_img):
    #     self.cont_img = cont_img

    def get_cont_img(self):
        return self.__cont_img

    # def set_style_img(self, style_img):
    #     self.style_img = style_img

    def get_style_img(self):
        return self.__style_img


task_collectors: dict[int, TaskCollector] = {}
tasks_table: dict[str, ChatProgress] = {}
task_collectors_lock: Lock = Lock()
table_lock: Lock = Lock()


def create_collector(chat_id, media_group_id):
    #if not chat_id in task_collectors:
    task_collectors[chat_id] = TaskCollector(media_group_id)
    return task_collectors[chat_id]


async def start_job(state):
    async with task_collectors_lock:
        content_img = copy.deepcopy(state.get_cont_img())
        style_img = copy.deepcopy(state.get_style_img())

    task_id = str(uuid.uuid4())
    await executor.add_task(task_id,
                            neural_style_transfer.ContentStylePair(('content.jpg', content_img), ('style.jpg', style_img)))

    # image_id = (await executor.task_ids())[0]
    # image_progress = await executor.get_progress(image_id)
    # cards = []
    #
    # async with states_lock:
    #     percent = image_progress[0] if image_progress[0] > 0 else 0
    #     cur_iter = percent / 100.0 * iters_num
    #     card = {
    #         "image_id": image_id,
    #         "percent": percent,
    #         "cur_iter": cur_iter,
    #         "iters_num": iters_num
    #     }
    #
    #     cards.append(card)
    #     res_img = copy.deepcopy(image_progress[1])

    #return res_img, cards, task_uuid
    return task_id


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


# @dp.message(Command("content", prefix="/"))
# async def command_content_handler(message: Message) -> None:
#     """
#     This handler receives messages with `/content` command
#     """
#     chat_id = message.chat.id
#     async with states_lock:
#         get_state(chat_id).set_mode(BotState.CONTENT)
#
#     await message.answer(f"Upload the {html.bold('content')} image, please")
#
#
# @dp.message(Command("style", prefix="/"))
# async def command_style_handler(message: Message) -> None:
#     """
#     This handler receives messages with `/style` command
#     """
#     chat_id = message.chat.id
#     async with states_lock:
#         get_state(chat_id).set_mode(BotState.STYLE)
#
#     await message.answer(f"Upload the {html.bold('style')} image, please")


# @dp.message()
# async def handle_albums(messages: Union[List[types.Message], types.Message], other_arg):
#     """This handler will receive a complete album of any type."""
#     pass
    # media_group = types.MediaGroup()
    # for obj in album:
    #     if obj.photo:
    #         file_id = obj.photo[-1].file_id
    #     else:
    #         file_id = obj[obj.content_type].file_id
    #
    #     try:
    #         # We can also add a caption to each file by specifying `"caption": "text"`
    #         media_group.attach({"media": file_id, "type": obj.content_type})
    #     except ValueError:
    #         return await message.answer("This type of album is not supported by aiogram.")
    #
    # await message.answer_media_group(media_group)


from aiogram_media_group import media_group_handler, MediaGroupFilter

class NotTwoPhotosException(Exception):
    pass


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


###############################################

#@dp.message()
# async def img_handler(message: Message) -> None:
#     """
#     Handler will forward receive a message back to the sender
#
#     By default, message handler will handle all message types (like a text, photo, sticker etc.)
#     """
#     try:
#         # Send a copy of the received message
#         if message.photo and message.media_group_id is not None:
#             chat_id = message.chat.id
#             group_id = message.media_group_id
#             message_id = message.message_id
#
#             file = await bot.get_file(message.photo[-1].file_id)
#             result = await bot.download_file(file.file_path)
#
#             with result as f:
#                 res_np = np.frombuffer(f.read(), np.uint8)
#                 img_np = cv2.imdecode(res_np, cv2.IMREAD_COLOR)
#             img_np = img_np[:, :, ::-1]  # BGR -> RGB
#
#             task_collector = None
#             do_processing = False
#             async with task_collectors_lock:
#                 if chat_id not in task_collectors.keys() or task_collectors[chat_id].media_group_id != group_id:
#                     # This is the second (style) photo of a media group
#                     # Ignoring the old collector object (if any), creating a new one
#                     task_collector = create_collector(chat_id, group_id)
#                     task_collector.add_image(message_id, img_np)
#                 elif task_collectors[chat_id] is not None and task_collectors[chat_id].media_group_id == group_id:
#                     # This is the first (content) proto of a media group
#                     # Saving it to the collector
#                     task_collector = task_collectors[chat_id]
#
#
#
#                     # if task_collector.get_style_img() is not None:
#                     #     raise Exception  # This is the THIRD photo
#
#                     task_collector.add_image(message_id, img_np)
#
#                     assert task_collector.get_cont_img() is not None and task_collector.get_style_img() is not None, "Logic error"
#
#                     do_processing = True
#                     await message.answer("Processing has started. Please, wait...")
#                     #res_img, prog_data, task_id = await start_job(task_collector)
#
#                 else:
#                     raise ValueError(f"Something strange happened!")
#
#             if do_processing:
#                 task_id = await start_job(task_collector)
#                 async with task_collectors_lock:
#                     del task_collectors[chat_id]  # The task is set, the collector is free
#                 async with table_lock:
#                     tasks_table[task_id] = ChatProgress(chat_id)
#
#         else:
#             raise Exception
#     except:
#         traceback.print_exc()
#         await message.answer(f"To start a job please send me two pictures {html.italic('in a single message')} - one for the {html.bold('content')} and one for the {html.bold('style')}")


# @dp.message()
# async def send_photo(call: CallbackQuery):
#     photo = InputFile('./data/content-images/tubingen.png')
#     await call.message.answer_photo(photo)


#@dp.message()
#async def send_photo(message: Message):
#    result: Message = await bot(SendPhoto(...))

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
