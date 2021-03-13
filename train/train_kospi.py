import requests, time
from bs4 import BeautifulSoup
from stock.services import market_data_service

FINANCE_KOSPI_URL = 'https://finance.naver.com/sise/sise_index_day.nhn?code='

def make_kospi_data(code, page): #NAVERD 증시 페이지에서 투자자 데이터를 적재
    finance_url = FINANCE_KOSPI_URL + code + '&page='
    total_kospi = []
    kospi_list = []

    for i in reversed(range(1,page)):
        time.sleep(0.5)
        url = finance_url + str(i)
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        date = soup.find_all("td", class_=["date"])
        kospi = soup.find_all("td", class_=["number_1"])

        # print(len(date))
        kospi_len = (int)(len(kospi)/4)
        for num in range(0, kospi_len):
            kospi_dict = {'date':date[num].text,
                           'kospi':kospi[num*4].text}
            kospi_list.append(kospi_dict)

        for num in range(len(kospi_list)):
            total_kospi.append(kospi_list.pop())

    for num in range(len(total_kospi)):
        print(total_kospi[num])

    market_data_service.remove_kospi()
    market_data_service.insert_kospi(total_kospi)

if __name__ == '__main__':
    '''
    종목코드
    051910 : LG CHEM = 137 page
    068270 : CELT
    005930 : SAMSUNG ELE
    '''

    code = 'KOSPI'
    page = 10
    
    #데이터 처음 만들기
    make_kospi_data(code, page)

    # res - investment_service.get_data(code)
    # for var in res:
    #     print(var)