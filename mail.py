from smtplib import SMTP
from email.header import Header
from email.mime.text import MIMEText

def main():
    sender = '2226010430@qq.com'
    receivers = '2657530327@qq.com'
    message = MIMEText('用python发送邮件的示例代码.','plain','utf-8')
    message['From'] = Header('GJX','utf-8')
    message['To'] = Header('ZZK','utf-8')
    message['Subject'] = Header('示例代码实验邮件’，‘utf-8')
    smtper = SMTP('smtp.126.com')
    # 请自行修改下面的登录口令
    smtper.login(sender,'mqtqdozlkqkheafg')
    smtper.sendmail(sender,receivers,message.as_string())
    print('邮件发送完成！')

if __name__ == '__main__':
    main()
