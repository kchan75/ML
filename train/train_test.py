import requests, time
from bs4 import BeautifulSoup
from stock.services import investment_service

FINANCE_KOSPI_URL = 'https://finance.naver.com/sise/sise_index.nhn?code=KOSPI'

def make_news_data(): #NAVERD 증시 페이지에서 뉴스 데이터를 적재
    finance_url = FINANCE_KOSPI_URL
    REFERER = 'https://finance.naver.com/sise/sise_index.nhn?code=KOSPI'
    AGENT = 'Mozilla/5.0'
    url = finance_url
    res = requests.get(url, headers={'referer': REFERER, 'User-Agent': AGENT})
    soup = BeautifulSoup(res.content, 'html.parser')
    kospi_value = soup.find_all("em", id=["now_value"])[0].contents[0]
    kospi_value = float(kospi_value.replace(",",""))
    print(kospi_value)

if __name__ == '__main__':
    #데이터 처음 만들기

    # make_news_data()

    listA=[1,2,3]
    print(listA[0:2])
    print(listA[1:3])
    listA[0:2] = listA[1:3]
    listA[2] = 4
    print(listA)
    print(sum(listA))
