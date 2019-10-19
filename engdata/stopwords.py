##불용어 제거
#swords=open(r'C:\Users\Z\Desktop\NI\한국어불용어100.txt', encoding='UTF8').read()
#stop_words=re.findall('[가-힣]+',swords)
def eng_stopwords(nar_df):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english') # NLTK로부터 불용어를 받아옵니다.
    
    tokenized_doc = nar_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])


    return(nar_df)
# 불용어를 제거합니다.
