# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:53:16 2019

@author: Z
"""
import pandas as pd
import re

# set column name as 0, for conveniency

def txt_to_df(path):
    
    data=open(path, encoding='cp949').read()   ### check encoding 
    
    data=data.split('\n')
    droplist=[]
    
    #### 텍스트만 추출
    for i in range(len(data)):
        if (re.compile('[a-z]+').search(data[i]) == None):
            droplist.append(i)
    droplist.reverse()
    
    for i in droplist:
        data.pop(i)
        
    data.pop(0)
    ######
    
    nar_df = pd.DataFrame(data)
    # 특수 문자 제거
    nar_df['clean_doc'] = nar_df[0].str.replace("[^a-zA-Z]", " ") ###주의, 여기는 column 이름이 0!!
    # 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
    nar_df['clean_doc'] = nar_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    # 전체 단어에 대한 소문자 변환
    nar_df['clean_doc'] = nar_df['clean_doc'].apply(lambda x: x.lower())
    
    return nar_df
