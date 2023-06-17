'''
this module contains functions and constants to send messages
via lab scripts telegram bot
'''


import subprocess
import traceback

import requests
import rich
from rich.markdown import Markdown

TOKEN = '5961004234:AAGgpbuWwnSc382mXL14m97Glu96_z2nBng'
SEND_URL = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
CHAT_ID = '258382605'


def send_message(message: str, silent: bool = True):
    '''
    send telegram message from lab_scripts bot.
    ** -- bold text, __ -- italics
    '''
    message = ('`' + traceback.extract_stack()[-2]
               .filename.rsplit('/', 1)[1] + '`: ' + message)
    answer = requests.post(SEND_URL,
                           json={'chat_id': CHAT_ID,
                                 'text': message,
                                 'parse_mode': 'Markdown'},
                           timeout=120)
    if answer.json()['ok']:
        if not silent:
            rich.print(Markdown(message))
    else:
        raise requests.exceptions.RequestException('message not sent.',
                                                   answer.json())
    return answer.json()


def run_or_send_error(cmd: str, msg: str, **kwargs) -> bool:
    '''
    wrapper to run shell commands and send telegram message
    with stderr if error occures

    cmd - command (as string)
    msg - message to send if error (will be bold and followed by stderr output)
    '''
    try:
        subprocess.run(cmd, shell=True, check=True, **kwargs)
        return True
    except subprocess.CalledProcessError as e:
        res = subprocess.run(cmd, shell=True, check=False, capture_output=True)
        # send_message('*' + msg + ':*\n`' +
        #              res.stderr.decode("utf-8") + '`', silent=True)
        rich.print(e, '\n')
        rich.print(res.stderr.decode("utf-8"))
        return False
