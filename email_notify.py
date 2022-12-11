#!/usr/bin/python3

import smtplib
import socket
import argparse
import os


def main():
    args=obtain_arguments()
    subject, body = write_email(args.system_name, args.step, args.theme)
    send_email(subject, body)

def obtain_arguments():
    parser = argparse.ArgumentParser(description='MD status notifier')
    parser.add_argument('system_name')
    parser.add_argument('step', help='current step: "rlx" or "md"')
    parser.add_argument('theme', help='theme of email: "error" or "success"')
    args = parser.parse_args()
    return args

def write_email(system_name, step, theme):
    host = socket.gethostname()
    subject = f'update about {system_name} calculation on {host}'

    if step == 'rlx':
        step = 'relaxation'
    elif step == 'md':
        step = 'molecular dynamics'
    else:
        raise ValueError('step should be "rlx" or "md"')

    if theme == 'error':
        body = f'calculation of {system_name} trajectory was interrupted during {step}'
    elif theme == 'success':
        body = f'{step} of {system_name} trajectory successfully completed'
    else:
        raise ValueError('theme should be "success" or "error"')

    return (subject, body)


def send_email(subject, body, to=['2601074@gmail.com']):
    gmail_user = 'fiyefiyefiye@gmail.com'
    gmail_password = 'odixypwjovmzqtzu'

    sent_from = gmail_user
    recipients = ', '.join(to)
    email_text = f'From: {sent_from}\nTo: {recipients}\nSubject: {subject}\n\n{body}'

    try:
        smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        smtp_server.ehlo()
        smtp_server.login(gmail_user, gmail_password)
        smtp_server.sendmail(sent_from, to, email_text)
        smtp_server.close()
        print("email sent successfully!")
    except Exception as ex:
        print("something went wrong...", ex)


if __name__ == "__main__":
    main()
