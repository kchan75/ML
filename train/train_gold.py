import requests, time
from bs4 import BeautifulSoup
from stock.services import market_data_service
import re

FINANCE_GOLD_URL = 'https://finance.naver.com/marketindex/worldDailyQuote.nhn?marketindexCd=CMDT_GC&fdtc=2'

def make_gold_data(page): #NAVERD 증시 페이지에서 투자자 데이터를 적재
    finance_url = FINANCE_GOLD_URL + '&page='
    total_gold = []
    gold_list = []

    for i in reversed(range(1,page)):
        time.sleep(0.5)
        url = finance_url + str(i)
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        date = soup.find_all("td", class_=["date"])
        gold = soup.find_all("td", class_=["num"])

        # print(date)
        # print(gold)
        gold_len = (int)(len(gold)/3)
        for num in range(0, gold_len):
            gold_dict = {'date':re.sub('\t|\n','',date[num].text),
                         'gold':re.sub('\t|\n','',gold[num*3].text)}
            gold_list.append(gold_dict)

        for num in range(len(gold_list)):
            total_gold.append(gold_list.pop())

    # for num in range(len(total_gold)):
    #     print(total_gold[num])

    market_data_service.remove_gold()
    market_data_service.insert_gold(total_gold)

if __name__ == '__main__':
    page = 10
    
    #데이터 처음 만들기
    make_gold_data(page)