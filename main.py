import telebot
from bot_logic import gen_pass, gen_emodji, flip_coin  # Импортируем функции из bot_logic
from config import TOKEN
from bot_logic import detect_games
# Замени 'TOKEN' на токен твоего бота
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я твой Telegram бот. Напиши команду /hello, /bye, /pass, /emodji или /coin  ")

@bot.message_handler(commands=['hello'])
def send_hello(message):
    bot.reply_to(message, "Привет! Как дела?")

@bot.message_handler(commands=['bye'])
def send_bye(message):
    bot.reply_to(message, "Пока! Удачи!")

@bot.message_handler(commands=['pass'])
def send_password(message):
    password = gen_pass(10)  # Устанавливаем длину пароля, например, 10 символов
    bot.reply_to(message, f"Вот твой сгенерированный пароль: {password}")

@bot.message_handler(commands=['emodji'])
def send_emodji(message):
    emodji = gen_emodji()
    bot.reply_to(message, f"Вот эмоджи': {emodji}")

@bot.message_handler(commands=['coin'])
def send_coin(message):
    coin = flip_coin()
    bot.reply_to(message, f"Монетка выпала так: {coin}")

@bot.message_handler(content_types=['photo'])
def ai_vision(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('/')[-1]
    downloaded_file = bot.download_file(file_info.file_path)
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)
    label_patch = "labels.txt"
    model_patch = "keras_model.h5"
    class_name, confidence_score = detect_games(model_patch, label_patch, file_name)
    if class_name == "strategy\n":
        class_name = "стратегия"
        info = "жанр игр где придется подумать"
    elif class_name == "actions\n":
        class_name = "экшен"
        info = "игры где нужна ловкость и быстрость к своим действиям"
    answer = f"это {class_name}, {info}, я в этом уверен на {int(confidence_score * 100)}%"
    bot.reply_to(message, answer)
# Запускаем бота
bot.polling()