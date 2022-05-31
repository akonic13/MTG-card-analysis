'''
Import needed libraries
'''
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
from random import randint

'''
Function: card_name_set:
    Uses selenium driver to access cardkingdom site 
    then uses soup object to pull all card names and sets
    from the site
'''
def card_name_set(URL):
    names_clean = []
    
    final_name1 = []
    sets_clean = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    sleep(randint(1,2))
    names = soup.find_all('div',class_='card-profile')
    sets = soup.find_all('span',class_='prints-current-set-name')
    
    for i,_ in enumerate(list(names)):
        names_clean.append(_.find_all('span','card-text-card-name'))
        
    for _ in names_clean:
        n = ''
        for t,i in enumerate(_):
            if t > 0:
                n += '//'+i.text.strip()
            else:
                n += i.text.strip()
            
        final_name1.append(n)
            
        
    for i,_ in enumerate(list(sets)):
        sets_clean.append(_.text.strip().split()[-1][1:4])
        
    return np.array(final_name1,dtype='object'), np.array(sets_clean,dtype='object')

'''
Function: card_types:
    Uses selenium driver to access cardkingdom site
    then uses soup object to pull all card types from site
'''
def card_types(URL):
    types1_clean = []
    types2_clean = []
    types = []
    dbl_sided = []
    Color = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    sleep(randint(1,2))
    cards = soup.find_all('div',class_='card-profile')
    
    for _ in cards:
        types.append(_.find_all('p',class_='card-text-type-line'))
        
    for _ in types:
        ColorInd = ''
        for i,t in enumerate(_):
            if i == 0:
                if t.text.strip().find('\n') != -1:
                    ColorInd += t.text.strip()[:t.text.strip().find('\n')]
                    if t.text.strip().find(chr(8212)) != -1:
                                           types1_clean.append(t.text.strip()[t.text.strip().find('\n')+1:t.text.strip().find(chr(8212))-1].replace(' ',''))
                    else:
                        types1_clean.append(t.text.strip()[t.text.strip().find('\n')+1:].replace(' ',''))
                        
                else:
                    ColorInd += ''
                    if t.text.strip().find(chr(8212)) != -1:
                                           types1_clean.append(t.text.strip()[t.text.strip().find('\n')+1:t.text.strip().find(chr(8212))-1].replace(' ',''))
                    else:
                        types1_clean.append(t.text.strip()[t.text.strip().find('\n')+1:].replace(' ',''))
                    
                        
                        
            if i != 0:
                dbl_sided.append('yes')
                if t.text.strip().find('\n') != -1:
                    ColorInd += t.text.strip()[:t.text.strip().find('\n')]
                    if t.text.strip().find(chr(8212)) != -1:
                                           types2_clean.append(t.text.strip()[t.text.strip().find('\n')+1:t.text.strip().find(chr(8212))-1].replace(' ',''))
                    else:
                        types2_clean.append(t.text.strip()[t.text.strip().find('\n')+1:].replace(' ',''))
                        
                else:
                    ColorInd += ''
                    if t.text.strip().find(chr(8212)) != -1:
                                           types2_clean.append(t.text.strip()[t.text.strip().find('\n')+1:t.text.strip().find(chr(8212))-1].replace(' ',''))
                    else:
                        types2_clean.append(t.text.strip()[t.text.strip().find('\n')+1:].replace(' ',''))
                        
            if len(_) == 1:
                types2_clean.append(np.nan)
                dbl_sided.append('no')
        Color.append(ColorInd)

    
    return np.array(types1_clean,dtype='object'),np.array(types2_clean,dtype='object'),np.array(dbl_sided,dtype='object'),np.array(Color,dtype='object')

'''
Function: mana_costs:
    Uses selenium driver to access cardkingdom site 
    then uses soup object to pull all mana costs as strings 
    then converts them to float CMC valeus
'''
def mana_cost(URL):
    manas = []
    mc = []
    mc_2 = []
    cmc = []
    phyrexian = []

    soup = BeautifulSoup(driver.page_source,'html.parser')
    sleep(randint(1,2))
    cost = soup.find_all('div',class_='card-profile')
        
    for _ in cost:
        manas.append(_.find_all('span',class_='card-text-mana-cost'))
        
    for _ in manas:
        gen_mana = 0
        mc5 = ''
        phyrx = 'no'
        for i in _:
            if i.text.strip().find('P') != -1:
                phyrx = 'yes'
            mc1 = i.text.strip().replace('{','')
            mc2 = mc1.replace('}','')
            mc3 = mc2.replace('/','')
            mc4 = mc3.replace('P','')
            mc5 += mc4
        
        mc.append(mc5)
        phyrexian.append(phyrx)
        
        for _ in mc:
            gen_mana = 0
            for i in _:
                if i.isdigit():
                    gen_mana += float(i)
                if i == 'X':
                    gen_mana += 0
                if (i != 'X') and (type(i)==str) and (i.isdigit()==False):
                    gen_mana += 1
                
        cmc.append(gen_mana)
        
    for _ in mc:
        if _ == '':
            mc_2.append(np.nan)
        else:
            mc_2.append(_)
        
    
    return np.array(mc,dtype='object'), np.array(cmc), np.array(phyrexian),np.array(mc_2,dtype='object')

'''
Function: color_ident:
    Takes in mana cost array and color indicators 
    as input and determines the color of the card
'''
def color_ident(castcost,colorinds):
    colors = []
    final_color = []
    for _ in castcost:
        color = ''
        color1 = ''
        color2 = ''
        color3 = ''
        color4 = ''
        color5 = ''
        if 'W' in set(_):
            color1 = 'W'
        if 'U' in set(_):
            color2 = 'U'
        if 'B' in set(_):
            color3 = 'B'
        if 'R' in set(_):
            color4 = 'R'
        if 'G' in set(_):
            color5 = 'G'
        color = color1+color2+color3+color4+color5
        
        if color == '':
            color = ''
            
        colors.append(color)
        
    for i,_ in enumerate(colorinds):
        _ = _.replace('Color Indicator:','')
        _ = _.replace('and','')
        _ = _.replace(',','')
        _ = _.replace(' ','')
        _ = _.replace('White','W')
        _ = _.replace('Blue','U')
        _ = _.replace('Black','B')
        _ = _.replace('Red','R')
        _ = _.replace('Green','G')
          
        final_color.append(''.join(dict.fromkeys(colors[i]+_)))
        
    for i,_ in enumerate(final_color):
        if _ == '':
            final_color[i]  = 'colorless'
    
    return np.array(final_color)
'''
Function: rarity:
    Uses selenium driver to access cardkingdom site
    then uses soup object to pull all string rarity 
    codes
'''
def rarity(URL):
    rarity_clean = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    sleep(randint(1,2))
    rarity = soup.find_all('span',class_='prints-current-set-details')
        
    for _ in rarity:
        rarity_clean.append(_.text.strip().split()[2][0])
            
    return np.array(rarity_clean)

'''
Function: price:
    Uses selenium driver to access cardkingdom site
    then uses soup object to pull price info as float
    from site
'''
def price(URL):
    prices_clean = []
    cards = []
    pricer = []
    soup = BeautifulSoup(driver.page_source,'html.parser')
    sleep(randint(1,2))
    cards = soup.find_all('div',class_='card-profile')        
    
    for _ in cards:
        pricer.append(_.find_all('a',class_='currency-usd'))
        
    for _ in pricer:
        pholder = []
        for i in _:
            p = i.text.strip()
            pholder.append(float(p[p.find('$')+1:]))
        
        if pholder == []:
            prices_clean.append(np.nan)
        else:
            prices_clean.append(np.array(pholder).min())

    return np.array(prices_clean,dtype='object')
    
'''
Initialize driver and dummy variables
Run for loop to go through each page of scryfall site 
Calls each function to pull all card info
Combines all info into a pandas dataframe and save as a csv file
'''
driver = webdriver.Chrome()
URL = 'https://scryfall.com/search?as=full&order=name&page='

lastpage = 1144
names = []
sets1 = []
types = []
types2 = []
two_sides = []
mana_cost1 = []
mana_cost2 = []
CMC1 = []
colorinds = []
phyrx1 = []
rarities = []
prices = []
for i in range(1,lastpage):
    driver.get(URL+str(i)+'&q=legal%3Acommander&unique=cards')
    
    card_names, set1=card_name_set(URL)
    names.extend(card_names)
    sets1.extend(set1)
                 
    card_type,card_type2,dbl_side,cols = card_types(URL)
    types.extend(card_type)
    types2.extend(card_type2)
    two_sides.extend(dbl_side)
    colorinds.extend(cols)
    
    mana_costs,CMC,phyrxs,mana_costs2 = mana_cost(URL)
    mana_cost1.extend(mana_costs)
    mana_cost2.extend(mana_costs2)
    CMC1.extend(CMC)
    phyrx1.extend(phyrxs)
    
    card_rarity = rarity(URL)
    rarities.extend(card_rarity)
    
    card_price = price(URL)
    prices.extend(card_price)
    
card_color = color_ident(mana_cost1,colorinds)

df = pd.DataFrame({'Card_Name':names,'Set':sets1,'Card_type1':types,'Card_type2':types2,'Modal':two_sides,'Mana_Cost':mana_cost2,'CMC':CMC1,'Phyrexian_Mana':phyrx1,'Color':card_color,'Rarity':rarities,'Price':prices})
df.to_csv('card_data.csv',index=False)
driver.close()
