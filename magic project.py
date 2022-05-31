'''
Import needed libraries
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from random import randint
import re

'''
Function: card_name:
    Uses selenium driver to access cardkingdom site 
    then uses soup object to pull all card names from site
'''
def card_name(URL):
    names_clean = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    names = soup.find_all('span',class_='productDetailTitle')
    
    for i,_ in enumerate(list(names)):
        names_clean.append(_.text.strip())
        
    return np.array(names_clean)

'''
Function: card_types:
    Uses selenium driver to access cardkingdom site
    then uses soup object to pull all card types from site
'''
def card_types(URL):
    types_clean = []
    final_types = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    names = soup.find_all('div',class_='productDetailType')
    
    for i,_ in enumerate(list(names)):
        types_clean.append(_.text.strip())
            
    for _ in types_clean:
        if re.search(r'[a-zA-Z]+',_.replace(' ','')) != None:
                final_types.append(re.search(r'[a-zA-Z]+',_.replace(' ',''))[0])
        else:
            final_types.append(np.nan)
    
    return np.array(final_types)

'''
Function: mana_costs:
    Uses selenium driver to access cardkingdom site 
    then uses soup object to pull all mana costs as strings 
    then converts them to float CMC valeus
'''
def mana_cost(URL):
    manas = []
    manas2 = ''
    costs = []
    final_cost = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    cost = soup.find_all('div',class_='productDetailCastCost')
        
    for _ in cost:
        manas.append(_.find_all('img'))
            
    for _ in manas:
        manas2 = ''
        for i in _:
            manas2 += i.get('alt')
                
        costs.append(manas2)
        
    for _ in costs:
          if re.search(r'\d',_)!=None:
              gen_mana = int(re.search(r'\d',_)[0])
              final_cost.append(len(_)+gen_mana-1)
          else:
              gen_mana = np.nan
              final_cost.append(gen_mana)
    
    return np.array(costs), np.array(final_cost)

'''
Function: rarity:
    Uses selenium driver to access cardkingdom site
    then uses soup object to pull all string rarity 
    codes
'''
def rarity(URL):
    rarity_clean = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    rarity = soup.find_all('div',class_='productDetailSet')
        
    for _ in rarity:
        rarity_clean.append(_.text.strip()[-2])
            
    return np.array(rarity_clean)

'''
Function: price:
    Uses selenium driver to access cardkingdom site
    then uses soup object to pull price info as float
    from site
'''
def price(URL):
    prices_clean = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    prices = soup.find_all('span','stylePrice')
            
    for _ in prices:
        prices_clean.append(float(_.text.strip()[1:]))
            
    return np.array(prices_clean[::4])
    
'''
Initialize driver and dummy variables
Run for loop to go through each page of cardkingdom site 
Calls each functionto pull all card info
Combines all info into a pandas dataframe
'''
driver = webdriver.Chrome()
URL = 'https://www.cardkingdom.com/catalog/search?search=mtg_advanced&filter%5Bsort%5D=name&filter%5Bsearch%5D=mtg_advanced&filter%5Btab%5D=mtg_card&filter%5Bname%5D=&filter%5Bedition%5D=&filter%5Bformat%5D=commander&filter%5Btype_mode%5D=any&filter%5Bcard_type%5D%5B10%5D=&filter%5Bpow1%5D=&filter%5Bpow2%5D=&filter%5Btuf1%5D=&filter%5Btuf2%5D=&filter%5Bconcast1%5D=&filter%5Bconcast2%5D=&filter%5Bprice_op%5D=&filter%5Bprice%5D=&filter%5Boracle_text%5D=&filter%5Bmanaprod_select%5D=any&page='
lastpage = 401

names = []
types = []
mana_cost1 = []
CMC1 = []
rarities = []
prices = []
for i in range(1,401):
    driver.get(URL+str(i))
    sleep(randint(2,10))
    card_names=card_name(URL)
    names.extend(card_names
                 )
    card_type = card_types(URL)
    types.extend(card_type)
    
    mana_costs,cmc = mana_cost(URL)
    mana_cost1.extend(mana_costs)
    CMC1.extend(cmc)
    
    card_rarity = rarity(URL)
    rarities.extend(card_rarity)
    
    card_price = price(URL)
    prices.extend(card_price)

df = pd.DataFrame({'Card_Name':names,'Card_type':types,'Mana_Cost':mana_cost1,'CMC':CMC1,'Rarity':rarities,'Price':prices})

driver.close()