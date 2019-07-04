---
layout: post
title: "[cheat sheet] SQL cheating sheet"
categories:
  - 코딩cheat sheet
tags:
  - SQL
  - cheating sheet
comment: true
---
{:toc}

### 시작하기 앞서

PostgreSQL을 가지고 배우지만, 여기서 배우는 모든 sql구문은 any major type of SQL Dababase(MySQL, Orcale등)에 모두 적용가능한 구문이다.

대소문자에 insensitive. 상관없지만, SQL keyword는 대문자로 쓰는게 관습.

#### Select

SELECT column1,column2,.., FROM table_name; (콤마로 여러개를 부른다. ';'가 나와야 문장이 끝난것으로 인식한다)

실무에선 SELECT *****를 쓸일은 거의 없음. 넘 큰데이터에 모든 정보를 다부르는거니까 속도가 무지 느려질것이다.

> SELECT first_name,last_name FROM actor;
>
> SELECT * FROM actor;

#### Select distinct

SELECT DISTINCT column1,column2,.., FROM table_name; 

중복값 없애고 부르기

> SELECT DISTINCT release_year from film;

#### Select where

특정 조건을 만족하는 데이터만을 부름

SELECT column1,column2,.., FROM table_name WHERE conditions;

> SELECT e_mail From customer where first_name='Jared' AND last_name = 'Rice';

#### Count

SELECT  COUNT(columns) FROM table; (count는 null값은 세지 않는다)

SELECT  COUNT(DISTINCT columns) FROM table; (이렇게 distinct의 갯수세기.)

#### Limit

몇줄보내줄지 정하는거.

>  SELECT * FROM customer LIMIT 5;

#### Order By

SELECT column1, column2 FROM table_name ORDER BY column1 ASC

> SELECT first_name, last_name from customer ORDER BY first_name ASC, last_name DESC; (first_name으로 오름차순sorting하고, first_name이 겹치는 사람들에 대해선 last_name으로 내림차순sorting한번 더하기)

#### Between, In, Like (Where안에 쓰이는애들)

**Between**

'value $$\ge$$ low and value $$\le$$ high'=='value BETWEEN low and high'

단순히 조건문을 더 fancy하게 만들어줌. (NOT BETWEEN도 가능)

> SELECT * FROM payment WHERE  amount NOT BETWEEN 8 AND 9 ; (8~9가 아닌 모든애들)

**IN**

value IN (value1,value2). 

value IN (SELECT value FROM table_name). (즉, IN의 평가대상이될 괄호 안에는 subquery가 들어갈수도 있다.) between과 마찬가지로, NOT IN도 가능

> SELECT customer_id,return_date FROM rental WHERE customer_id IN (10,100);

**LIKE**

> SELECT first_name, last_name FROM customer WHERE first_name LIKE 'Jen%'; (Jen으로 시작하는 패턴과 맞는애들을 반환해라.) 

패턴매칭에 대해선 뒤에서 더 자세히.

% : 암거나 관계없이 **무수히** 받을 수잇음

_ : 암거나 관계없이 '**하나**' 받을 수 있음

> SELECT first_name, last_name FROM customer WHERE first_name LIKE '_her%'; (Cheryl, Theresa,..)	  

### Aggregate Function

사실 count도 이 중 하다. 여러row들을 signale value로 합쳐주는 애들.

**MIN** : SELECT MIN(amount) FROM payment;

**MAX** : SELECT MAX(amount) FROM payment;

**AVG** : SELECT ROUND(AVG(amount),3) FROM payment; (ROUND는 3자리까지 허용하는 우리가 아는 그 함수.)

**SUM** : SELECT SUM(amount) FROM payment;

#### GROUP BY (중요!)

SELECT colum1, aggregate_func(column2) FROM table_name GROUP BY column1; (GROUP BY로 묶이는 대상의 컬럼은, select에서 불러준다. 안해도되는 sql도 있지만 안정성을 위해.)

group을 묶어서 각 group마다 aggregate func를 취하는 형태로 많이 쓰인다. 물론 aggregate func없이도 사용가능 (판다스의 groupby상기)

> SELECT customer_id, SUM(amount), COUNT(amount) FROM payment GROUP BY customer_id;
>
> SELECT rating, COUNT(rating) FROM film GROUP BY rating; (각 rating의 count를 하기 위해선 이렇게. 각 count가 어느 rating의 count인지를 알려주기 위해 rating도 select한다.)

#### Having

주로 group by와 함께 조건을 추가하기 위해 쓰임. (group by의 결과물 전용where같은 느낌)

SELECT column1,agg_func(column2) FROM table_name GROUP BY column1 HAVING condition

where은 groupby전에, having은 groupby한 대상에 대해 쓰이기에, **select, from, where, groupby, having**순으로 쓰인다.

> SELECT custormer_id, SUM(amount) FROM payment GROUP BY customer_id HAVING SUM(amount) > 200;
>
> (groupby에만 쓰일수 있는 sum이 having에 들간것 주의. 저런 condi를 만족하는 group만 반환해달라.)

#### As

그냥 별명짓기

SELECT customer_id, SUM(amount) AS my_name1, AVG(amount) AS my_name2 FROM payment GROUP BY customer_id;

SELECT cus.customer_id, first_name, payment_id FROM customer as cus INNER JOIN payment as ppay ON ppay.customer_id=cus.customer_id;

이렇게 테이블에도 별명 지어줄수도

### JOIN

다수의 테이블을 합칠때. (주로 테이블A의 primary key를 테이블B의 외래키와 비교해서 합치는 경우이다.)

#### Inner Join

교집합의 데이터 반환.

SELECT A.prim_key, A.col11 B.forin_key, B.col21 FROM A INNER JOIN B ON A.prim_key = B.forein_key; (사실 select에서 'A.' 과 같은 테이블 명시는 col이 겹칠때만 해도됨. 즉 col11이 A에만 있음 A.col11대신 col11로 해도된다.)

<img width="207" alt="inner_join" src="https://user-images.githubusercontent.com/31824102/60664516-220f6880-9e9d-11e9-865f-ed228363c3b1.PNG">

> SELECT customer.customer_id, first_name, payment_id FROM customer 
> INNER JOIN payment ON payment.customer_id=customer.customer_id;
>
> (여기서 payment_id는 payment table에만 있었다.)

#### full outer join

전체 집합. 이럼 당근 A에는 있었지만 B에는 없던 데이터가 있을것. 그건 그냥 'null'로 셀이 채워져서 반환된다

SELECT * FROM tableA FULL OUTER JOIN tableB ON tableA.name=tabelB.name;

<img width="191" alt="full_outer_join" src="https://user-images.githubusercontent.com/31824102/60664514-2176d200-9e9d-11e9-8b17-91e1bd5f4990.PNG">

#### Left outer join (right outer도 마찬가지)

From의 대상이 left임. left에 있는 애들 중심으로 데려오기.

SELECT * FROM tableA LEFT OUTER JOIN tableB ON tableA.name=tabelB.name; (LEFT JOIN만써줘도 된다.)

<img width="207" alt="left_outer_join" src="https://user-images.githubusercontent.com/31824102/60664517-220f6880-9e9d-11e9-9269-25be8b0f86e9.PNG">

#### 덧. Left outer join with WHERE (left에만 있는애들 고르기)

SELECT * FROM tableA LEFT OUTER JOIN tableB ON tableA.name=tabelB.name WHERE tableB.name IS null; (A에만 있는 애들은 B에서 null로 나올것이기에, 이렇게 where로 pure A를 구할 수 있다.)

<img width="206" alt="left_outer_join_where" src="https://user-images.githubusercontent.com/31824102/60664513-2176d200-9e9d-11e9-9053-6cdd8b8b890f.PNG">

#### 덧2. full outer join with WHERE

SELECT * FROM tableA FULL OUTER JOIN tableB ON tableA.name=tabelB.name WHERE tableA.name IS null OR tableB.name IS null;  

<img width="201" alt="full_outer_join_where" src="https://user-images.githubusercontent.com/31824102/60664515-2176d200-9e9d-11e9-88f9-6b7e3e733a15.PNG">





[join정리한 사이트](https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/)

#### Union

key그런거 상관없이, 그냥 같은크기면 concat해주는 애.(duplicate면 하나의 row로 합침.)

SELECT col1,col2 FROM tbl_name1 **UNION** SELECT col1,col2 FROM tbl_name2;

(합치는 두개가 같은 수의column이어야. col1끼리, col2끼리 데이터타입이 같아야.)

> SELECT * FROM customer_list1 UNION SELECT * FROM customer_list2;

---

## Advanced SQL Commands

#### Timestamp

시간(혹은 달력)의 의미를 보존한 timestamp object를 다루는법.

이부분은 Mysql등에서 조금씩 다를 수 있다. 쓰기전 document를 봐라. (ex. 'mysql datetime')

date '2001-09-28' + integer '7'같은 timestamp용 operator나 age(timestamp_obj) 같이 timestamp용 function을 쓸 수 있다.

> SELECT SUM(amount), extract(month from payment_date) AS mmonth FROM payment GROUP BY mmonth; (datetime obj에서 '달'만 떼와서 활용한 extract예시)

#### Mathmatical function

모든 func는 'postgresql math'등 검색. 우리가 흔히 아는 math func임

#### String function and operators

역시나 'postgresql string function'검색.

SELECT first_name &#124;' '&#124; last_name AS full_name FROM customer; (이름과 성을 concat.)

SELECT lower(first_name) FROM customer;

**지원하지 않는 더 복잡한 형태는 정규표현식(regular expression)을 찾아봐라.**

#### Subquery

쿼리안에 들어가있는 쿼리. 괄호()로 감싸져표현됨.

간단하게 말하면 select를 두번 쓸 수 있게해줌. 즉 훨씬더 간결하고 유연한쿼리작성 가능.

SELECT film_id, rental_rate FROM film WHERE rental_rate > **( SELECT AVG(rental_rate) from film)**;

SELECT film_id FROM film 
WHERE film_id IN (SELECT...); (subquery로 list를 만든경우)

#### Self Join

같은 테이블내의 데이터(row)를 합치고 싶은 경우 self join을 함. 이 경우 left와 right를 구분하기 위해 AS 를 써서 alias를 붙임. (sql면접에서 많이 나온다고 한다.)

SELECT a.first_name,a.last_name,b.first_name,b.last_name 
FROM customer as a, customer as b
WHERE a.first_name = b.last_name;

본인과 합치는것이기에, 실제 JOIN구문을 안쓰고 할수도 있다.

이렇게 JOIN을 쓰는 구문도 있다(같은 결과다).

SELECT a.first_name,a.last_name,b.first_name,b.last_name 
FROM customer as a JOIN customer as b
ON a.first_name = b.last_name;

---

참고한 자료

시각화로 join 잘 설명한 사이트 : https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/
