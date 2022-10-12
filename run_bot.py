import telebot
import os
import subprocess

cwd = os.getcwd()
API_KEY = '5707667195:AAGQ9CTKn1O6oOGHvCT5AhL_RZUzb4fgbkU'
infer_file = cwd+'/infer_from_url.py'

bot = telebot.TeleBot(API_KEY)
@bot.message_handler(commands=['sum'])
def sum(message):
    # cm_url = 'python /content/bert-extractive-summarization/infer_from_url.py -url '+'https://dantri.com.vn/the-gioi/nga-don-dap-doi-ten-lua-vao-ukraine-20221011200011293.htm'
    # os.system(cm_url)
    #bot.reply_to(message, 'hello')
    bot.send_message(message.chat.id, 'Please wait...')
    proc = subprocess.Popen(['python', infer_file,  '-url', message.text[5:]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    rep = proc.communicate()[0][1152:]
    bot.send_message(message.chat.id, rep)
bot.infinity_polling(timeout=10, long_polling_timeout = 5)