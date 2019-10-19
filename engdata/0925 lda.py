import pandas as pd
import re
import numpy 

#dataset=pd.read_csv(r'C:\Users\Z\Desktop\NI\Identity (community) Narrative Coding.csv', encoding='cp949')
# dataset.drop('Score', axis=1, inplace=True)

data=open(r'C:\Users\Z\Desktop\NI\engdata\Narratives_valid324.txt', encoding='cp949').read()

data=data.split('\n')
droplist=[]

for i in range(len(data)):
    if (re.compile('[a-z]+').search(data[i]) == None):
        droplist.append(i)
droplist.reverse()

for i in droplist:
    data.pop(i)
    
data.pop(0)

nar_df = pd.DataFrame(data)
# 특수 문자 제거
nar_df['clean_doc'] = nar_df[0].str.replace("[^a-zA-Z]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
nar_df['clean_doc'] = nar_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 전체 단어에 대한 소문자 변환
nar_df['clean_doc'] = nar_df['clean_doc'].apply(lambda x: x.lower())


#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#sid = SentimentIntensityAnalyzer()
#for sent in lines_list:
#   ss = sid.polarity_scores(sent)
#print(ss['compound'])
 
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#analyser = SentimentIntensityAnalyzer()

#def sentime
#nt_analyzer_scores(sentence):
#    score = analyser.polarity_scores(sentence)
#    print("{:-<40} {}".format(sentence, str(score)))
#score = analyser.polarity_scores(sentence)


#nar_df[0][1].replace(';',' ').replace(',',' ').split().split('.')
nar_df['parsed_0']=''
for i in range(len(nar_df[0])):
    text=nar_df[0][i]
    parsed_0=text.replace('\n','bazinga').replace('.','bazinga').replace('\t','bazinga').split('bazinga')
    nar_df['parsed_0'][i]=parsed_0


############## sentence length

#sen_len=[]    
#for i in nar_df['parsed_0']:
#    sen_len.append(len(i))
#numpy.mean(sen_len)
#########################

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
nar_df['bscore-fscore']=''
nar_df['pos_score']=''
nar_df['neu_score']=''
nar_df['neg_score']=''
nar_df['compound_score']=''


#nar_df['compound_score']=''
for i in range(len(nar_df['parsed_0'])):
    scores=[]
    pos_scores=[]; neu_scores=[]; neg_scores=[]; compound_scores=[]
    
    for j in nar_df['parsed_0'][i]:
        score=sid.polarity_scores(j)
        scores.append(score)
        pos_scores.append(score['pos']); neg_scores.append(score['neg']); neu_scores.append(score['neu'])
        compound_scores.append(score['compound'])
    
    front_score=numpy.mean(compound_scores[0:len(compound_scores)//2])
    back_score=numpy.mean(compound_scores[len(compound_scores)//2:])
    
    nar_df['compound_score'][i]= numpy.mean(compound_scores)
    nar_df['pos_score'][i]= numpy.mean(pos_scores); nar_df['neu_score'][i]= numpy.mean(neu_scores); nar_df['neg_score'][i]= numpy.mean(neg_scores)
    nar_df['bscore-fscore'][i]= back_score-front_score
    


##########################
    
    
#from nltk.corpus import sentiwordnet as swn

#for i in nar_df['parsed_0'][1]:
#breakdown = swn.senti_synset(nar_df['parsed_0'][1][1])


######################3
##불용어 제거
#swords=open(r'C:\Users\Z\Desktop\NI\한국어불용어100.txt', encoding='UTF8').read()
#stop_words=re.findall('[가-힣]+',swords)

from nltk.corpus import stopwords
stop_words = stopwords.words('english') # NLTK로부터 불용어를 받아옵니다.

tokenized_doc = nar_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
# 불용어를 제거합니다.

#####################################
#freq plot

tokenized_doc[:5]

import collections

all_tokens=[]
for i in tokenized_doc:
    all_tokens.extend(i)


c2 = collections.Counter(all_tokens)
print(c2.most_common(3))

####
from matplotlib import pyplot as plt
 
plt_x=[]
plt_y=[]
for i in c2.most_common(100):
    plt_x.append(i[0])
    plt_y.append(i[1])

plt.bar(plt_x,plt_y, width=0.03, color="purple")
plt.show()

plt.plot(plt_y)

############### Wordcloud

import matplotlib
from wordcloud import WordCloud
from PIL import Image


#font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

#font_path = font_path,

wc_mask = numpy.array(Image.open(r'C:\Users\Z\Desktop\NI\conversation.jpg'))

wc = WordCloud(#width=1600, height=1600,
               background_color="white",max_font_size=200,
               mask=wc_mask,
               max_words=2000, colormap=matplotlib.cm.inferno)

wordcloud=wc.generate_from_frequencies(dict(c2.most_common(200)))


plt.figure( figsize=(30,30), facecolor='k' )
plt.tight_layout(pad=0)
plt.axis("off")
plt.imshow(wc)

#plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
#plt.axis("off")
#plt.show()
####################빈도분석



from gensim import corpora
dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1]) # 수행된 결과에서 두번째 뉴스 출력. 첫번째 문서의 인덱스는 0


import gensim
NUM_TOPICS = 5 #20개의 토픽, k=20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
    
    
for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic 비율은',topic_list) 
    
    
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

pyLDAvis.show(vis) ####### display 말고!
pyLDAvis.save_html(vis)
print('hello')
    
    
def make_topictable_per_doc(ldamodel, corpus, texts):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)
    

topictable = make_topictable_per_doc(ldamodel, corpus, tokenized_doc)
topictable = make_topictable_per_doc(ldamodel, corpus, tokenized_doc)
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
topictable[:10]


dataset['가장 비중이 높은 토픽'] = pd.Series(topictable['가장 비중이 높은 토픽'])
