#! py -3
# -*- coding: utf-8 -*-

"""
阿里云VPS
tf训练结束提示
邮件接口
"""

import datetime
import smtplib
from email.header import Header
from email.mime.text import MIMEText


def send_email(email_content):
    user = "yuwancumiana@sina.cn"
    password = "mq2020."
    receivers = ['2506930314@qq.com']
    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText(email_content+"""
    \n
    请访问<a href="http://149.129.113.178:6006" target="_blank">http://149.129.113.178:6006</a>查看训练数据。
    """, 'html', 'utf-8')
    message['From'] = Header(user)
    message['To'] = Header("锤爷", 'utf-8')

    subject = '您的TensorFlow训练已结束'
    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtp = smtplib.SMTP_SSL()
        smtp.connect("smtp.sina.com", 465)
        smtp.login(user, password)
        smtp.sendmail(user, receivers, message.as_string())
        smtp.quit()
        print(datetime.datetime.now(), ">> 邮件发送成功")
    except smtplib.SMTPException as e:
        print(datetime.datetime.now(), ">> Error: 发送邮件失败", e)


if __name__ == '__main__':
    send_email("发送邮件提示")
