import os
os.chdir(r'C:\Users\Z\Desktop\NI\code')


import txt_to_df
import parse_and_sen
from stopwords import eng_stopwords


path=r'C:\Users\Z\Desktop\NI\engdata\Narratives_valid324.txt'

nar_df = txt_to_df.txt_to_df(path); nar_df = parse_and_sen.parse(nar_df)
# sen scores
nar_df = parse_and_sen.sentiment(nar_df)
#stopwords
nar_df = eng_stopwords(nar_df)

##############################3

import nltk
from nltk.tag import pos_tag
tagged_list = pos_tag(nar_df[0][1])

nltk.word_tokenize(nar_df[0][1])
pos_tag(
nltk.word_tokenize(nar_df[0][2])
)

nar_df['parsed_0']=''
for i in range(len(nar_df[0])):
    text=nar_df[0][i]
    parsed_0=text.replace('\n','bazinga').replace('.','bazinga').replace('\t','bazinga').split('bazinga')
    nar_df['parsed_0'][i]=parsed_0
