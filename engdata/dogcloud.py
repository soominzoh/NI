from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

from wordcloud import WordCloud, STOPWORDS

##text = pd.read_csv(r'C:\Users\Z\Desktop\zhm\whatcat.csv', encoding='cp949')
text = open(r'C:\Users\Z\Desktop\zhm\dog.txt').read()
#text=re.sub('whatcat', '',text)

# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
alice_mask = np.array(Image.open(r'C:\Users\Z\Desktop\zhm\cat.jpg'))

font_path=(r'C:\Users\Z\Desktop\BM.ttf')

wc = WordCloud(background_color="white", max_words=10, mask=alice_mask,
               font_path=font_path,
               contour_width=30,
    contour_color='steelblue')

# generate word cloud
wc.generate(text)

# store to file
#wc.to_file(path.join(d, "alice.png"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure(figsize=(20,20))
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()

#####


import matplotlib
from wordcloud import WordCloud
from PIL import Image


#font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

#font_path = font_path,
import numpy
import collections

wc_mask = numpy.array(Image.open(r'C:\Users\Z\Desktop\zhm\dog.png'))

wc = WordCloud(width=00, height=400,background_color="white",max_font_size=2000,
               #mask=wc_mask, 
               font_path=font_path,
               max_words=2000, colormap=matplotlib.cm.hsv)

#wordcloud=wc.generate_from_frequencies(dict(c2.most_common(200)))



c2 = collections.Counter(text.split('\n'))

wordcloud=wc.generate_from_frequencies(dict(c2.most_common(200)))



plt.figure( figsize=(20,20), facecolor='k' )
plt.tight_layout(pad=0)
plt.axis("off")
plt.imshow(wc)

#plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
#plt.axis("off")
#plt.show()
