import asyncio
import copy
import logging
import sys
import traceback
import uuid
from asyncio import Lock

import cv2
import numpy as np


from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, InputFile, CallbackQuery, BufferedInputFile
from aiogram.methods import SendPhoto
from magic_filter import MagicFilter

import neural_style_transfer
from task_executor import Executor


# Bot token can be obtained via https://t.me/BotFather
TOKEN = "7433346137:AAF2vjCKBNK_WlXJKBR1_7qFIWN4G5KExyE"

# Initialize Bot instance with default bot properties which will be passed to all API calls
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()

content_weight = 1e1
style_weight = 1e5
tv_weight = 0e3
optimizer = 'lbfgs'
model = 'vgg19'
init_method = 'content'
levels_num = 2
iters_num = 800

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


# Bot states
class BotState:
    STYLE = "style"
    CONTENT = "content"

    def __init__(self):
        self.value = BotState.CONTENT
        self.cont_img = None
        self.style_img = None

    def set_mode(self, state):
        self.value = state

    def get_mode(self):
        return self.value

    def set_cont_img(self, cont_img):
        self.cont_img = cont_img

    def get_cont_img(self):
        return self.cont_img

    def set_style_img(self, style_img):
        self.style_img = style_img

    def get_style_img(self):
        return self.style_img




states: dict[int, BotState] = {}
tasks_table: dict[str, ChatProgress] = {}
states_lock: Lock = Lock()
table_lock: Lock = Lock()


def get_state(chat_id):
    if not chat_id in states:
        states[chat_id] = BotState()
    return states[chat_id]


async def process_st(state):
    async with states_lock:
        content_img = copy.deepcopy(state.get_cont_img())
        style_img = copy.deepcopy(state.get_style_img())

    task_uuid = str(uuid.uuid4())
    await executor.add_task(task_uuid,
                            neural_style_transfer.ContentStylePair(('content.jpg', content_img), ('style.jpg', style_img)))

    image_id = (await executor.task_ids())[0]
    image_progress = await executor.get_progress(image_id)
    cards = []

    async with states_lock:
        percent = image_progress[0] if image_progress[0] > 0 else 0
        cur_iter = percent / 100.0 * iters_num
        card = {
            "image_id": image_id,
            "percent": percent,
            "cur_iter": cur_iter,
            "iters_num": iters_num
        }

        cards.append(card)
        res_img = copy.deepcopy(image_progress[1])

    return res_img, cards, task_uuid


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
                         f"You can now send me two pictures - one for the content and one for the style")


@dp.message(Command("content", prefix="/"))
async def command_content_handler(message: Message) -> None:
    """
    This handler receives messages with `/content` command
    """
    chat_id = message.chat.id
    async with states_lock:
        get_state(chat_id).set_mode(BotState.CONTENT)

    await message.answer(f"Upload the {html.bold('content')} image, please")


@dp.message(Command("style", prefix="/"))
async def command_style_handler(message: Message) -> None:
    """
    This handler receives messages with `/style` command
    """
    chat_id = message.chat.id
    async with states_lock:
        get_state(chat_id).set_mode(BotState.STYLE)

    await message.answer(f"Upload the {html.bold('style')} image, please")


@dp.message()
async def img_handler(message: Message) -> None:
    """
    Handler will forward receive a message back to the sender

    By default, message handler will handle all message types (like a text, photo, sticker etc.)
    """
    try:
        # Send a copy of the received message
        if message.photo:
            file = await bot.get_file(message.photo[-1].file_id)
            result = await bot.download_file(file.file_path)
            #print(sys.getsizeof(result))
            with result as f:
                res_np = np.frombuffer(f.read(), np.uint8)
                img_np = cv2.imdecode(res_np, cv2.IMREAD_COLOR)
            #print(img_np.shape)
            img_np = img_np[:, :, ::-1] # BGR -> RGB

            chat_id = message.chat.id
            do_processing = False

            async with states_lock:
                state = get_state(chat_id)
                cur_mode = state.get_mode()
                if cur_mode == BotState.CONTENT:
                    state.set_cont_img(img_np)
                elif cur_mode == BotState.STYLE:
                    state.set_style_img(img_np)
                else:
                    raise ValueError(f"Unknown mode {cur_mode}")

                if state.get_cont_img() is not None and state.get_style_img() is not None:
                    do_processing = True

            if do_processing:
                res_img, prog_data, task_id = await process_st(state)
                async with table_lock:
                    tasks_table[task_id] = ChatProgress(chat_id)
    except:
        traceback.print_exc()
        await message.answer("Nice try!")


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
