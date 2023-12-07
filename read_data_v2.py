import json
import pandas as pd


df = pd.DataFrame()

useful_tot = 50000
non_useful_tot = 50000
useful_idx = 0
non_useful_idx = 0
idx = 0
file = open("yelp_academic_dataset_review.json", 'r', encoding='utf-8')
for line in file.readlines():
    if non_useful_idx==non_useful_tot and useful_idx==useful_tot:
        break
    else:
        dic = json.loads(line)
        if dic['useful']==0 and non_useful_idx<non_useful_tot:
            df = pd.concat([df, pd.DataFrame([dic])])
            non_useful_idx+=1
            idx+=1
        elif dic['useful']>=10 and useful_idx<useful_tot:
            df = pd.concat([df, pd.DataFrame([dic])])
            useful_idx+=1
            idx+=1
        else:
            idx+=1
            continue

print('total lines read: ', idx)
print('total lines useful read: ', useful_idx)
df.to_csv(r'E:/dev/review_useful_data.csv')



df = pd.DataFrame()

tot_0_10 = 10000
idx = 0
file = open("yelp_academic_dataset_review.json", 'r', encoding='utf-8')
for line in file.readlines():
    if idx==tot_0_10:
        break
    else:
        dic = json.loads(line)
        if dic['useful']<10 and dic['useful']>0:
            df = pd.concat([df, pd.DataFrame([dic])])
            idx+=1
        else:
            idx+=1
            continue

print('total lines read: ', idx)
df.to_csv('review_useful_data_0_to_10.csv')


df = pd.DataFrame()
number = 100000
idx = 0
file = open("yelp_academic_dataset_review.json", 'r', encoding='utf-8')
for line in file.readlines():
    if idx==number:
        break
    else:
        dic = json.loads(line)
        df = pd.concat([df, pd.DataFrame([dic])])
        idx+=1

df.to_csv('review.csv')


start_idx = 3500000


df = pd.DataFrame()
number = 100000
idx = 0
star_1_idx =star_2_idx=star_3_idx=star_4_idx=star_5_idx=0
star_1_tot =star_2_tot=star_3_tot=star_4_tot=star_5_tot= 50
file = open("yelp_academic_dataset_review.json", 'r', encoding='utf-8')
for line in file.readlines():
    if idx<start_idx:
        idx+=1
        continue
    if star_1_idx==star_1_tot and star_2_idx==star_2_tot and star_3_idx==star_3_tot and star_4_idx==star_4_tot\
            and star_5_idx==star_5_tot:
        break
    else:
        dic = json.loads(line)
        if dic['stars']==1 and star_1_idx<star_1_tot:
            df = pd.concat([df, pd.DataFrame([dic])])
            star_1_idx += 1
        elif dic['stars']==2 and star_2_idx<star_2_tot:
            df = pd.concat([df, pd.DataFrame([dic])])
            star_2_idx += 1
        elif dic['stars']==3 and star_3_idx<star_3_tot:
            df = pd.concat([df, pd.DataFrame([dic])])
            star_3_idx += 1
        elif dic['stars']==4 and star_4_idx<star_4_tot:
            df = pd.concat([df, pd.DataFrame([dic])])
            star_4_idx += 1
        elif dic['stars']==5 and star_5_idx<star_5_tot:
            df = pd.concat([df, pd.DataFrame([dic])])
            star_5_idx += 1
        else:
            continue

df.to_csv('review_final_test.csv')