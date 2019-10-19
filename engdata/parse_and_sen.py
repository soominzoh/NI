import re
import numpy

def parse(nar_df):

    nar_df['parsed_0']=''
    
    for i in range(len(nar_df[0])):
        text=nar_df[0][i]
        parsed_0=text.replace('\n','bazinga').replace('.','bazinga').replace('\t','bazinga').split('bazinga')
        nar_df['parsed_0'][i]=parsed_0
        
    nar_df=nar_df
    return (nar_df)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def sentiment(nar_df):
     
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
        
    
    return nar_df