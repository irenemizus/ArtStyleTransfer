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
from task_executor import executor, iters_num

# Bot token can be obtained via https://t.me/BotFather
TOKEN = "7433346137:AAF2vjCKBNK_WlXJKBR1_7qFIWN4G5KExyE"

# Initialize Bot instance with default bot properties which will be passed to all API calls
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()


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
states_lock: Lock = Lock()


def get_state(chat_id):
    if not chat_id in states:
        states[chat_id] = BotState()
    return states[chat_id]


async def process_st(state):
    async with states_lock:
        content_img = copy.deepcopy(state.get_cont_img())
        style_img = copy.deepcopy(state.get_style_img())

    await executor.add_task(str(uuid.uuid4()),
                            neural_style_transfer.ContentStylePair(('content.jpg', content_img), ('style.jpg', style_img)))
    await executor.run()
    image_id = (await executor.task_ids())[0]
    image_progress = await executor.get_progress(image_id)

    async with states_lock:
        progress = image_progress[0] if image_progress[0] > 0 else 0
        cur_iter = progress / 100.0 * iters_num
        prog_data = [progress, cur_iter, iters_num]
        res_img = copy.deepcopy(image_progress[1])

    return res_img, prog_data


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
                res_img, prog_data = await process_st(state)
                new_img_np = res_img[:, :, ::-1] # RGB -> BGR

                new_img_np = np.clip(new_img_np * 255, 0, 255).astype('uint8')
                _, new_res = cv2.imencode('.jpg', new_img_np)
                new_res_bytes = new_res.tobytes()
                #with open("content.jpg", "wb") as f:
                #    f.write(new_res_bytes)

                bif = BufferedInputFile(new_res_bytes, 'tmp.jpg')
                #print(new_res.shape)
                await message.answer_photo(bif)
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

async def main() -> None:
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
