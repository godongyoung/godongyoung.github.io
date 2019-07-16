---
layout: post
title: "[cheat sheet] Open API 활용방법 정리(with Python) "
categories:
  - 코딩cheat sheet
tags:
  - cheating sheet
  - open api
  - REST
  - python
comment: true
---
{:toc}


버스 실시간 도착정보 데이터를  open API로 불러올 것이다.

사용할 api의 주소 : https://www.data.go.kr/dataset/15000314/openapi.do

API에서 제공하는 서비스 리스트

- getArrInfoByRouteAllList - 경유노선 전체 정류소 도착예정정보를 조회한다

- getArrInfoByRouteList - 한 정류소의 특정노선의 도착예정정보를 조회한다

- getLowArrInfoByStIdList - 정류소ID로 저상버스 도착예정정보를 조회한다 

- getLowArrInfoByRouteList - 한 정류소의 특정노선의 저상버스 도착예정정보를 조회한다

## 자료 요청 방식 (step by step)

앞서 설명한대로, url을 호출하면 xml형태의 데이터를 반환해줍니다. 
입력해주어야 하는 요청파라미터는 서비스별로 정의되어 있으며(doc에 있음) 여기서는 **'특정 버스의 전체 정류장 정보'**를 불러올 것입니다. 
따라서 **인증키**와 **루트ID**가 필요합니다.

따라서 예시는 다음과 같습니다.

**예시 URL : http://ws.bus.go.kr/api/rest/arrive/getArrInfoByRouteAll?ServiceKey=인증키&busRouteId=100100118**

0. 필요한 모듈을 불러옵니다

```python
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
```

1. 요청파라미터로 넣어줄 값들을 설정합니다


```python
key = 'put_your_service_key_here' ### 여기에 신청한 본인의 key를 넣어야 합니다!
busRouteId='100100118'
```

2. doc에 적혀있는 end-point url과 나의 요청파라마터를 합해주어 최종적인 url을 만듭니다.


```python
queryParams = 'ServiceKey='+key+'&busRouteId='+busRouteId
url = 'http://ws.bus.go.kr/api/rest/arrive/getArrInfoByRouteAll?'+queryParams

print(url)
```

    http://ws.bus.go.kr/api/rest/arrive/getArrInfoByRouteAll?ServiceKey=put_your_service_key_here&busRouteId=100100118
3. 앞서 Import한 request의 get메소드를 이용하여 respond를 받습니다. (받아온 객체는 Response객체로, 활용을 위해선 추가의 처리를 필요로 합니다)


```python
req = requests.get(url) 
print(type(req))
```

    <class 'requests.models.Response'>
4. 이 response객체를 텍스트로 변환. 이제 다음과 같이 xml형태의 데이터를 string으로 확인할 수 있습니다.


```python
html = req.text 
print(type(html)) #이제 string객체로 바꼈다.
print(html[:150]) #이렇게 내용물을 확인할 수 있다.
```

    <class 'str'>
    <?xml version="1.0" encoding="UTF-8" standalone="yes"?><ServiceResult><comMsgHeader/><msgHeader><headerCd>0</headerCd><headerMsg>정상적으로 처리되었습니다.</heade

5. 아직 xml형태를 가진 문서를 string으로 바꾼것이기 때문에, 이 중 사용할 데이터를 잘끄집에 내야합니다. 이때 xml의 구조를 알아서 알아서 잘 파악하는 html parser가 잇습니다. 여기서는 Beautifulsoup의 html parser를 이용합니다.


```python
soup = BeautifulSoup(html, 'html.parser')
```

6. 이제 xml형태를 인식할 수 있게 됬기에, 내가 원하는 attribute를 find_all함수를 이용하여 불러올수 있습니다


```python
finded_values=soup.find_all('stnm')
[x.text for x in finded_values][:5] # .text를 통해 안의 내용물만을 불러옵니다
```


    ['선진운수종점', '구산동사거리', '한솔아파트입구선정중학교후문', '갈현동미미아파트', '선일여고입구']



## 이제 실제 예시 (Toy code)

이번에는 다양한 parameter값을 넣어 데이터를 불러오고, xml을 파싱한후, 판다스 dataframe으로 만드는 전체 과정에 대해 코드를 짜봅니다.

(**앞선 api는 서울시교통정보과에서 만든 api ([주소](https://www.data.go.kr/dataset/15000314/openapi.do))이고, 아래의 api는 [서울시열린데이터광장](http://data.seoul.go.kr/dataList/datasetView.do?infId=OA-12913&srvType=S&serviceKind=1&currentPageNo=2&searchValue=&searchKey=null)에서의 api이다. url의 연결방법이 매 사이트마다 조금 다르다는걸 유의하세요!**)

여기서는 각 정류장의 정보를 startnumber\~endnumber로 지정하여, 1\~2000번째 정류장을 for문을 이용하여 불러보고, 이를 판다스 DataFrame으로 바꾸고 저장합니다


```python
startnumber=1
endnumber=1000
CommerceInfor = {}

while endnumber <= 2000:
  print('getting data from %s to %s'%(startnumber,endnumber))
  url='http://openapi.seoul.go.kr:8088/put_your_service_key_here/xml/GetParkInfo/'+str(startnumber)+'/'+str(endnumber)+'/'

  req = requests.get(url)
  html = req.text
  soup = BeautifulSoup(html, 'html.parser')
  
  attr_to_find_list=['parking_code','parking_name','addr','parking_type','que_status','capacity','cur_parking','pay_yn','rates','add_rates']
  for each_attr in attr_to_find_list:
    finded_attr=soup.find_all(each_attr)
    if CommerceInfor.get(each_attr) is None:
      CommerceInfor[each_attr]=[x.text for x in finded_attr]
    else:
      CommerceInfor[each_attr]=CommerceInfor[each_attr]+[x.text for x in finded_attr]

  startnumber += 1000
  endnumber += 1000
    
print('end!')
df = pd.DataFrame(CommerceInfor)
```

    getting data from 1 to 1000
    getting data from 1001 to 2000
    end!

결과를 확인해 봅니다

```python
print(df.shape)
df.head()
```

```python
(2000, 10)
```

|      | parking_code | parking_name                | addr                 | parking_type | que_status | capacity | cur_parking | pay_yn | rates | add_rates |
| ---- | ------------ | --------------------------- | -------------------- | ------------ | ---------- | -------- | ----------- | ------ | ----- | --------- |
| 0    | 1369476      | 탑골공원 버스전용주차장(시) | 종로구 종로2가 38-4  | NS           | 1          | 2        | 2           | N      | 0     | 0         |
| 1    | 1369476      | 탑골공원 버스전용주차장(시) | 종로구 종로2가 38-4  | NS           | 1          | 2        | 2           | N      | 0     | 0         |
| 2    | 1384878      | 교육청길(구)                | 강북구 미아동 137-19 | NS           | 0          | 94       | 0           | Y      | 200   | 200       |
| 3    | 1384878      | 교육청길(구)                | 강북구 미아동 137-19 | NS           | 0          | 94       | 0           | Y      | 200   | 200       |
| 4    | 1384878      | 교육청길(구)                | 강북구 미아동 137-19 | NS           | 0          | 94       | 0           | Y      | 200   | 200       |

잘 작동합니다! time을 이용하여 다음과 같이 자동으로 저장하는 등의 활용을 해볼 수 있습니다

if time % 600==0:
  df.to_excel('auto_save' + str(cnt) + '.xlsx', sheet_name = 'sheet1')
