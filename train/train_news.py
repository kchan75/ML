import requests, time
from bs4 import BeautifulSoup
from stock.services import investment_service

FINANCE_NEWS_URL = 'https://finance.naver.com/news/news_list.nhn?mode=LSS3D&section_id=101&section_id2=258&section_id3=401'

def make_news_data(date): #NAVERD 증시 페이지에서 뉴스 데이터를 적재
    finance_url = FINANCE_NEWS_URL + '&date=' + date
    total_news = []
    news_list = []

    url = finance_url
    res = requests.get(url)

    soup = BeautifulSoup(res.content, 'html.parser')
    news = soup.find_all(["dd","dt"], class_=["articleSubject"])
    press = soup.find_all("span", class_=["press"])
    wdate = soup.find_all("span", class_=["wdate"])

    # print(news)
    # print(press)
    # print(wdate)

    for i in range(len(news)):
        print(news[i].text, press[i].text, wdate[i].text)

    #     print(len(info))
    #     info_len = (int)(len(info)/9)
    #     for num in range(0, info_len):
    #         news_dict = {'code':code,
    #                        'date':info[num*9].text,
    #                        'price':info[num*9+1].text,
    #                        'korea':info[num*9+5].text,
    #                        'foreign':info[num*9+6].text}
    #         news_list.append(news_dict)
    #
    #     for num in range(len(news_list)):
    #         total_news.append(news_list.pop())
    #
    # for num in range(len(total_news)):
    #     print(total_news[num])

    # investment_service.delete_data(code)
    # investment_service.insert_data(total_news)

if __name__ == '__main__':
    while True:
        print("TEST")
        time.sleep(2)

    date = '20200704'
    
    #데이터 처음 만들기
    make_news_data(date)

    # res - investment_service.get_data(code)
    # for var in res:
    #     print(var)