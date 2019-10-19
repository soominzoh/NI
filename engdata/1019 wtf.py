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
tagged_list = nltk.pos_tag(nar_df[0][1])

nltk.word_tokenize(nar_df[0][1])
pos_tag(
nltk.word_tokenize(nar_df[0][2])
)

nar_df['parsed_0']=''
for i in range(len(nar_df[0])):
    text=nar_df[0][i]
    parsed_0=text.replace('\n','bazinga').replace('.','bazinga').replace('\t','bazinga').split('bazinga')
    nar_df['parsed_0'][i]=parsed_0

###################################
    
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize

STANFORD_POS_MODEL_PATH = r"C:\Users\Z\Desktop\NI\tools\stanford-postagger-2018-10-16\stanford-postagger-2018-10-16\models\english-bidirectional-distsim.tagger"
STANFORD_POS_JAR_PATH = r"C:\Users\Z\Desktop\NI\tools\stanford-postagger-2018-10-16\stanford-postagger-2018-10-16\stanford-postagger.jar"

    
pos_tagger = StanfordPOSTagger(STANFORD_POS_MODEL_PATH, STANFORD_POS_JAR_PATH)
    

text = """Facebook CEO Mark Zuckerberg acknowledged a range of mistakes on Wednesday, 
including allowing most of its two billion users to have their public profile data scraped by outsiders. 
However, even as he took responsibility, he maintained he was the best person to fix the problems he created."""

tokens = word_tokenize(text)
print(tokens)
print()
print(pos_tagger.tag(tokens)) ### os home 지정해주자