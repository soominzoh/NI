

from konlpy.tag import Kkma  
kkma=Kkma()  
import numpy as np
import pandas as pd
import re
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt



dataset=pd.read_csv(r'C:\Users\Z\Desktop\NI\engdata\Raw_1.csv', encoding='cp949')
dataset.drop('Score', axis=1, inplace=True)


news_df = pd.DataFrame(dataset.Narrative)
# 특수 문자 제거
news_df['clean_doc'] = news_df['Narrative'].str.replace("[^가-힣]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
# 전체 단어에 대한 소문자 변환
### news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

######################3
##불용어 제거
swords=open(r'C:\Users\Z\Desktop\NI\한국어불용어100.txt', encoding='UTF8').read()
stop_words=re.findall('[가-힣]+',swords)


#######
    
dataset['morphed']=range(len(dataset.Narrative))

instance=Counter([])
for i in range(len(dataset.Narrative)):
    instance+=Counter(kkma.morphs(dataset.Narrative[i]))


tags=dict(instance)

tags_copy=tags

keys=tags_copy.keys()


for i in list(keys):
    if len(i) < 2:
        del tags[i]

######
wc = WordCloud(font_path="NanumGothic", width=1200, height=800,
scale=2.0, max_font_size=250)

gen = wc.generate_from_frequencies(tags)

plt.figure()
plt.imshow(gen, interpolation='bilinear')
#wc.to_file("korean.png")
#plt.close()
#plt.show()

tags_sorted=sorted(tags.items(),reverse=False, key=lambda t : t[1])

tags_sorted[1]


keys=[]
values=[]
for i in range(len(tags_sorted)):
    keys.append(tags_sorted[i][0])
    values.append(tags_sorted[i][1])



#####################3
    # 입력 데이터
plt.title('scatter')
#plt.bar(keys[1:1000],values[1:1000])
#plt.show()

plt.plot(values[1:4000])
plt.show()


