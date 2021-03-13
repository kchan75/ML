import requests, time
from bs4 import BeautifulSoup
from stock.services import market_data_service
import re

FINANCE_DOLLAR_URL = 'https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_USDKRW'

def make_dollar_data(page): #NAVERD 증시 페이지에서 투자자 데이터를 적재
    finance_url = FINANCE_DOLLAR_URL + '&page='
    total_dollar = []
    dollar_list = []

    for i in reversed(range(1,page)):
        time.sleep(0.5)
        url = finance_url + str(i)
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        date = soup.find_all("td", class_=["date"])
        dollar = soup.find_all("td", class_=["num"])

        # print(date)
        # print(dollar)
        dollar_len = (int)(len(dollar)/2)
        for num in range(0, dollar_len):
            dollar_dict = {'date':re.sub('\t|\n','',date[num].text),
                         'dollar':re.sub('\t|\n','',dollar[num*2].text)}
            dollar_list.append(dollar_dict)

        for num in range(len(dollar_list)):
            total_dollar.append(dollar_list.pop())

    # for num in range(len(total_dollar)):
    #     print(total_dollar[num])

    market_data_service.remove_dollar()
    market_data_service.insert_dollar(total_dollar)

if __name__ == '__main__':
    page = 10
    
    #데이터 처음 만들기
    make_dollar_data(page)