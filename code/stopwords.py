##불용어 제거
#swords=open(r'C:\Users\Z\Desktop\NI\한국어불용어100.txt', encoding='UTF8').read()
#stop_words=re.findall('[가-힣]+',swords)
def eng_stopwords(nar_df):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english') # NLTK로부터 불용어를 받아옵니다.
    
    tokenized_doc = nar_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

    nar_df['clean_doc']=tokenized_doc
    return(nar_df)
# 불용어를 제거합니다.


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

wc_mask = numpy.array(Image.open(r'C:\Users\Z\Desktop\NI\twitter_mask.png'))

wc = WordCloud(width=1600, height=1600,background_color="white",max_font_size=200,
               mask=wc_mask,
               max_words=2000, colormap=matplotlib.cm.inferno)

wordcloud=wc.generate_from_frequencies(dict(c2.most_common(200)))


plt.figure( figsize=(20,10), facecolor='k' )
plt.tight_layout(pad=0)
plt.axis("off")
plt.imshow(wc)
