import nltk; import konlpy

from nltk.tokenize import sent_tokenize

text="mama, just killed a man. put a gun against his head, pulled my trigger now he's dead. mama, life has just begun."
print(sent_tokenize(text))

from konlpy.tag import Okt  
okt=Okt()  
print(okt.morphs("sdaasㄹㅇㄴㄻㅇㄴ다"))

from konlpy.tag import Kkma  
kkma=Kkma()  
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

sequences=[['a', 1], ['b', 2], ['c', 3]] # 리스트의 리스트 또는 행렬 또는 뒤에서 배울 개념인 2D 텐서.
X,y = zip(*sequences) # *를 추가
print(X)
print(y)
help(zip())

####
import numpy as np  

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)


X, y = np.arange(10).reshape((5, 2)), range(5)
# 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
print(X)
print(list(y)) #레이블 데이터\

y=[10, 11, 12, 13, 15]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

########
import numpy as np
X, y = np.arange(0,24).reshape((12,2)), range(12)

n_of_train = int(len(X) * 0.8) # 데이터의 전체 길이의 80%에 해당하는 길이값을 구한다.
n_of_test = int(len(X) - n_of_train) # 전체 길이에서 80%에 해당하는 길이를 뺀다.
print(n_of_train)
print(n_of_test)

##################################
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
vocab=Counter() # 파이썬의 Counter 모듈을 이용하면 단어의 모든 빈도를 쉽게 계산할 수 있습니다.  

text=['A barber is a person.', 'a barber is good person.', 'a barber is huge person.', 'he Knew A Secret!', 'The Secret He Kept is huge secret.', 'Huge secret.', 'His barber kept his word.', 'a barber kept his word.', 'His barber kept his secret.', 'But keeping and keeping such a huge secret to himself was driving the barber crazy.', 'the barber went up a huge mountain.']

sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence=word_tokenize(i) # 단어 토큰화를 수행합니다.
    result = []

    for word in sentence: 
        word=word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                result.append(word)

                vocab[word]=vocab[word]+1 #각 단어의 빈도를 Count 합니다.

    sentences.append(result) 
print(sentences)

vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)
print(vocab_sorted)
####
word_to_index={}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
        i=i+1
        word_to_index[word]=i
print(word_to_index)

word_to_index['ass']=9

word_to_index={word : index+1 for index, word in enumerate(vocab)}


############################################3

from konlpy.tag import Okt  
okt=Okt()  
token=okt.morphs("나는 자연어 처리를 배운다")  

word2index={}
for voca in token:
     if voca not in word2index.keys():
       word2index[voca]=len(word2index)
print(word2index)

def one_hot_encoding(word, word2index):
       one_hot_vector = [0]*(len(word2index))
       index=word2index[word]
       one_hot_vector[index]=1
     
       return one_hot_vector


###########

