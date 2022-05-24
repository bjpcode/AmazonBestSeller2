#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install beautifulsoup4 pandas


# In[2]:


from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import re


# In[3]:


url = 'https://www.amazon.it/gp/bestsellers/electronics/473246031/ref=zg_bs_nav_ce_3_1497228031'


# In[4]:


request = Request(url, headers={'User-agent': 'Mozilla/5.0'})
html = urlopen(request)
soup = BeautifulSoup(html, 'html.parser')


# In[5]:


urls = []
for a_href in soup.find_all("a", class_ = "a-link-normal",tabindex = -1, href=True):
    urls.append(a_href["href"])
string = 'https://www.amazon.it'
urls = [string + x for x in urls]
urls


# In[15]:


title1 = []
RRP = []
promo = []
rank = []
seller = []
for url1 in urls:
    request1 = Request(url1, headers={'User-agent': 'Mozilla/5.0'})
    html1 = urlopen(request1)
    soup1 = BeautifulSoup(html1, 'html.parser')
    title1.append(soup1.find('span', id = "productTitle").get_text(strip=True))
    promo.append(soup1.find('span', class_ = "a-offscreen").get_text(strip=True))
    seller.append(soup1.find('div', class_ = "tabular-buybox-text a-spacing-none").get_text(strip=True))
    #RRP.append(soup1.find('span', class_="a-price a-text-price").get_text(strip=True))
    #rank.append(soup1.find('ul', class_="a-unordered-list a-nostyle a-vertical zg_hrsr").get_text(strip=True))


# In[16]:


d = {'Title': title1, 'price': promo, 'Seller': seller}#, 'price': price, 'URL': urls}#, 'price': price
df = pd.DataFrame(data=d)
df


# In[18]:


df.to_excel (r'C:\Users\baijinpeng\Desktop\意大利线上竞争2.xlsx', index = False, header=True)


# In[ ]:




