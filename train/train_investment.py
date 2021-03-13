import requests, time
from bs4 import BeautifulSoup
from stock.services import investment_service

FINANCE_INVEST_URL = 'https://finance.naver.com/item/frgn.nhn?code='

def make_invest_data(code, page): #NAVERD 증시 페이지에서 투자자 데이터를 적재
    finance_url = FINANCE_INVEST_URL + code + '&page='
    total_invest = []
    invest_list = []

    for i in reversed(range(1,page)):
        time.sleep(0.5)
        url = finance_url + str(i)
        res = requests.get(url)

        soup = BeautifulSoup(res.content, 'html.parser')
        num = soup.find_all("table", class_=["type2"])
        info = num[1].find_all("td", class_=["tc", "num"])
        print(info)

        print(len(info))
        info_len = (int)(len(info)/9)
        for num in range(0, info_len):
            invest_dict = {'code':code,
                           'date':info[num*9].text,
                           'price':info[num*9+1].text,
                           'korea':info[num*9+5].text,
                           'foreign':info[num*9+6].text}
            invest_list.append(invest_dict)

        for num in range(len(invest_list)):
            total_invest.append(invest_list.pop())

    for num in range(len(total_invest)):
        print(total_invest[num])

    investment_service.delete_data(code)
    investment_service.insert_data(total_invest)

if __name__ == '__main__':
    '''
    종목코드
    051910 : LG CHEM = 137 page
    068270 : CELT
    005930 : SAMSUNG ELE
    '''

    code = '005930'
    page = 190
    
    #데이터 처음 만들기
    make_invest_data(code, page)

    # res - investment_service.get_data(code)
    # for var in res:
    #     print(var)