import requests
import traceback

TOKEN = '5961004234:AAGgpbuWwnSc382mXL14m97Glu96_z2nBng'
SEND_URL = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
CHAT_ID = '258382605'


def send_message(message: str, silent: bool = False):
    '''
    send telegram message from lab_scripts bot.
    ** -- bold text, __ -- italics
    '''
    message = ('`' + traceback.extract_stack()[-2]
               .filename.rsplit('/', 1)[1] + '`: ' + message)
    answer = requests.post(SEND_URL,
                           json={'chat_id': CHAT_ID,
                                 'text': message,
                                 'parse_mode': 'Markdown'})
    if answer.json()['ok'] == True:
        if not silent:
            print(message)
    else:
        raise requests.exceptions.RequestException('message not sent.')
    return answer.json()
