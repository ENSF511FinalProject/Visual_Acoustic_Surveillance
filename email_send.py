## https://www.cnblogs.com/shenh/p/14267345.html
from smtplib import SMTP_SSL
from email.mime.text import MIMEText

def sendMail(message,Subject,sender_show,recipient_show,to_addrs,cc_show=''):
    '''
    :param message: str email content
    :param Subject: str email subject
    :param sender_show: str show sender
    :param recipient_show: str show recipient
    :param to_addrs: str sending address 
    :param cc_show: str show cc
    '''
    # sender's address and password
    user = 'xxx@gmail.com'
    password = 'xxx'
    
    # mail content
    msg = MIMEText(message, 'plain', _charset="utf-8")
    # mail subject
    msg["Subject"] = Subject
    # sender show
    msg["from"] = sender_show
    # recipient show
    msg["to"] = recipient_show
    # cc show
    msg["Cc"] = cc_show

    with SMTP_SSL(host="smtp.gmail.com",port=465) as smtp:
        # login
        smtp.login(user = user, password = password)
        # sned email
        smtp.sendmail(from_addr = user, to_addrs=to_addrs.split(','), msg=msg.as_string())

def alert_message(obj):
    ''' create alert message
    obj(str): alert object
    '''
    text = "This email is sent because we suspect {} from our device, you might want to take some action about it.".format(obj)
    return text
if __name__ =='__main__':
    message = alert_message("fire")
    Subject = 'Alert'

    # appear sender
    sender_show = 'xxx'
    # appear recipient
    recipient_show = 'xxx'
    # to address
    to_addrs = 'xxx.com'
    sendMail(message,Subject,sender_show,recipient_show,to_addrs)