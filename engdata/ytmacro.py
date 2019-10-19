from selenium import webdriver
import requests
import os; import re
import time

### id zsm21375test@gmail.com
### pw z95s9m25

driver=webdriver.Chrome(r'C:\Users\Z\Desktop\학교\youtubemacro\chromedriver')   
time.sleep(3)
driver.get(r'https://www.youtube.com')
time.sleep(1)

# 로그인부터 한 다음에, 시작 영상 지정하고, 그 다음에 go 쳐넣기

x = input()
if x=='go':
    print('dasd')
    
print('lets get started')

titlelist=[]
chlist=[]

num=0
while True:    
    title=driver.find_element_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[3]/div[1]/div/div[7]/div[2]/ytd-video-primary-info-renderer/div/h1/yt-formatted-string').text
    ch=driver.find_element_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[3]/div[1]/div/div[9]/div[3]/ytd-video-secondary-info-renderer/div/div[2]/ytd-video-owner-renderer/div[1]/ytd-channel-name/div/div/yt-formatted-string/a').text
    
    titlelist.append(title); chlist.append(ch)
    time.sleep(2) 
    
#따봉
    driver.find_element_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[3]/div[1]/div/div[7]/div[2]/ytd-video-primary-info-renderer/div/div/div[3]/div/ytd-menu-renderer/div/ytd-toggle-button-renderer[1]/a/yt-icon-button/button/yt-icon').click()
#다음 추천동영상 클릭
    driver.find_element_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[3]/div[1]/div/div[12]/ytd-watch-next-secondary-results-renderer/div[2]/ytd-compact-autoplay-renderer/div[2]/ytd-compact-video-renderer/div[1]/div[1]/a/h3/span').click()

    time.sleep(4) 
    
    ####텍스트로 100마다 저장
    if num%100==0:
        f = open(r"C:\Users\Z\Desktop\학교\youtubemacro\zsm_trial"+(str(num))+".txt",'w', encoding='UTF8')
        f.write(str(titlelist))
        f.write('\n');f.write('\n');f.write('\n')
        f.write(str(chlist))

        f.close()
    ####
    #팝업탭 닫기
    driver.window_handles
    while len(driver.window_handles)!=1:
        
        last_tab = driver.window_handles[-1]
        driver.switch_to.window(window_name=last_tab)
        
        driver.close()    
    
    ###
    num=num+1


################################
# after manual break #
######################
    
f = open(r"C:\Users\Z\Desktop\학교\youtubemacro\zsm_trial33.txt",'w', encoding='UTF8')
f.write(str(titlelist))
f.write('\n');f.write('\n');f.write('\n')
f.write(str(chlist))

f.close()
