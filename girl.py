from time import time 
from threading import Thread

import requests

class DownloadHanlder(Thread):

    def _init_(self,url):
        super()._init_()
        self.url = url
    def run(self):
        filename = self.url[self.url.rfind('/')+1:]
        resp = requests.get(self.url)
        with open('/Users/Hao/' +filename,'wb') as f:
            f.write(resp.content)

def main():
    resp = requests.get( 'http://api.tianapi.com/meinv/?key=e7ab8000819cf411d7eb3fc2a7c68545&num=10')
    data_model = resp.json()
    for mm_dict in data_model['newslist']:
        url = mm_dict['picUrl']
        DownloadHanlder(url).start()

if __name__  == '_main_':
    main()
