import MySQLdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts

#%% 讀取資料
rawdata = pd.read_csv("flights_sample_3m(19-23)_raw.csv", encoding="utf-8")
print(len(rawdata)) # 3百萬筆資料
print(len(rawdata.columns)) # 32欄位
''' <2019-2023 每日每趟航班延遲資料>
['FL_DATE'航班日期, 'AIRLINE'航空公司, 'AIRLINE_DOT'航司代碼, 'AIRLINE_CODE'航司IATA代碼, 'DOT_CODE'航司DOT代碼,
'FL_NUMBER'航班號, 'ORIGIN'出發機場IATA號碼, 'ORIGIN_CITY'出發城市名, 'DEST'抵達機場IATA號碼, 'DEST_CITY'抵達城市名,
'CRS_DEP_TIME'表訂起飛時間, 'DEP_TIME'實際起飛時間, 'DEP_DELAY'起飛延遲多少時間, 'TAXI_OUT'滑行出需時, 'WHEELS_OFF'機輪離地時間,
'WHEELS_ON'機輪著地時間, 'TAXI_IN'滑行入需時, 'CRS_ARR_TIME'表定抵達時間, 'ARR_TIME'實際抵達時間, 'ARR_DELAY'抵達延遲多少時間,
'CANCELLED'是(1)否(0)取消航班, 'CANCELLATION_CODE'航班取消類型, 'DIVERTED'是(1)否(0)轉降航班, 'CRS_ELAPSED_TIME'表定航班需時,
'ELAPSED_TIME'航班需時(taxi-fly-taxi), 'AIR_TIME'飛行時間, 'DISTANCE'飛行距離, 'DELAY_DUE_CARRIER'延遲歸因航空公司,
'DELAY_DUE_WEATHER'延遲歸因天氣, 'DELAY_DUE_NAS'延遲歸因航空系統, 'DELAY_DUE_SECURITY'延遲歸因安全,'DELAY_DUE_LATE_AIRCRAFT延遲歸因飛機']
#CANCELLATION_CODE: A code that indicates the carrier- reported cause for the cancellation. Permissible Values: A = Carrier; B = Weather; C = National Aviation System; D = Security;
'''


#%%資料前處理
#日期格式轉換: 使用pd.to_Datetime將日期字串轉換為日期變數,'%Y/%m/%d'指定日期字串格式
rawdata["FL_DATE"]=pd.to_datetime(rawdata["FL_DATE"],format="%Y/%m/%d")
#將日期變數拆分為年、月、日三個欄位，和年-月欄位。以利後面分析。
rawdata["Year"]=rawdata["FL_DATE"].dt.year
rawdata["Month"]=rawdata["FL_DATE"].dt.month
rawdata["Day"]=rawdata["FL_DATE"].dt.day
rawdata["Year-month"]=rawdata["FL_DATE"].dt.strftime("%Y-%m")
#計算2019~2023年全美國內航班數量變化趨勢
count2019=len(rawdata[rawdata["Year"]==2019])
count2020=len(rawdata[rawdata["Year"]==2020])
count2021=len(rawdata[rawdata["Year"]==2021])
count2022=len(rawdata[rawdata["Year"]==2022])
count2023pre=int(len(rawdata[rawdata["Year"]==2023])*1.5) #predict:695,226
count202308=len(rawdata[rawdata["Year"]==2023]) #as of 2023/08/31
#匯出2019~2023航班數量：再畫2019~2023pre航班數量
numbercount=pd.Series([count2019,count2020,count2021,count2022,count2023pre,count202308],index=['2019','2020','2021','2022','2023pre','202308'])
numbercount.to_csv("2019~2023 num of flight_online.csv")


#篩選在rawdata中,2022~2023年(8月)的資料，作為後續討論用。
filter1=rawdata["Year"]==2022
filter2=rawdata["Year"]==2023
data22_23=rawdata[filter1|filter2]
#篩選在data22_23中,欄位AIRLINE_CODE裡的聯合航空與其共享航班號的航司，UAs(IATA code): United Airline(UA), Mesa Airline(YV), SkyWest(OO), Republic Airline(YX)
filter3=data22_23["AIRLINE_CODE"]=='UA'
filter4=data22_23["AIRLINE_CODE"]=='YV'
filter5=data22_23["AIRLINE_CODE"]=='OO'
filter6=data22_23["AIRLINE_CODE"]=='YX'
data22_23_UAs=data22_23[filter3|filter4|filter5|filter6]
#統計在data22_23_UAs中,非準時航班比例(delay,cancel,divert) #使用df_delay.value_counts().get(True,0): 如果遇到True就算一個否則算0個
filter_delay=data22_23_UAs["ARR_DELAY"]>0
filter_can=data22_23_UAs["CANCELLED"]==1
filter_div=data22_23_UAs["DIVERTED"]==1
delay_counts=filter_delay.value_counts().get(True,0)
can_counts=filter_can.value_counts().get(True,0)
div_counts=filter_div.value_counts().get(True,0)
delay_counts_percent=((delay_counts/len(data22_23_UAs))*100).round(2)
can_counts_percent=((can_counts/len(data22_23_UAs))*100).round(2)
div_counts_percent=((div_counts/len(data22_23_UAs))*100).round(2)
normal_percent=(100-(delay_counts_percent+can_counts_percent+div_counts_percent)).round(2)
#ANS: normal_percent:64.69 delay_counts:32.79 can_counts:2.24 div_counts_percent:0.28 (百分比)
state_distribution=[normal_percent,delay_counts_percent,can_counts_percent,div_counts_percent]
statedf=pd.DataFrame(state_distribution,columns=["state"]).T
statedf.to_csv("航班狀態占比.csv")
#額外統計ARR_DEALY<15mins
arrdelay_yes=data22_23_UAs["ARR_DELAY"]>0
arrdelay_less15=data22_23_UAs["ARR_DELAY"]<15 #意即>15,na者刪除
input_ml_yesless15=data22_23_UAs[arrdelay_yes&arrdelay_less15] #即ARR中<15者
arrdelayall=data22_23_UAs[arrdelay_yes]
print('ARR_DELAY中多少比例是小於15分鐘: %.2f'%(len(input_ml_yesless15)/len(arrdelayall))) #0.44


#統計在data22_23_UAs中選取欄位ARR_DELAY,Year-month成為delay_Year_month,計算delay航班每個月的情形
delay_Year_month=data22_23_UAs.loc[:,["ARR_DELAY","Year-month"]]
filter_arrdelay_Year_month=delay_Year_month["ARR_DELAY"]>0 #filter概念: 選有ARR_DELAY時間>0者
#把filter_arrdelay_Year_month帶回原檔delay_Year_month，只保留arr dalay筆數和y-m
delay_Year_month=delay_Year_month[filter_arrdelay_Year_month] 
#groupby y-m，統計每個時間單位arr delay個數
delay_trend=delay_Year_month.groupby("Year-month").count() 
print(delay_trend)
delay_trend.to_csv("delay_trend_Year-month.csv") 

#%% 個案討論: 針對SFO起飛，UAs的資料
#選取回歸方程式的輸入資料，產生input_ml：ORIGIN:SFO，AIRLINE_CODE:UA,OO
filter_SFO=data22_23_UAs['ORIGIN']=="SFO"
input_ml=data22_23_UAs[filter_SFO]
#只保留回歸方程式必要欄位
'''
DEST'抵達機場IATA號碼, 
DEP_DELAY'起飛延遲多少時間, 
TAXI_OUT'滑行出需時, 
TAXI_IN'滑行入需時, 
ARR_DELAY'抵達延遲多少時間, -->應變數
AIR_TIME'飛行時間,
DISTANCE'飛行距離, 
DELAY_DUE_CARRIER延遲歸因航空公司,
DELAY_DUE_WEATHER延遲歸因天氣, 
DELAY_DUE_NAS'延遲歸因航空系統,
DELAY_DUE_SECURITY'延遲歸因安全,
DELAY_DUE_LATE_AIRCRAFT延遲歸因飛機,
'''
input_colneed=input_ml.loc[:,["DEST","DEP_DELAY","TAXI_OUT","TAXI_IN","ARR_DELAY","AIR_TIME","DISTANCE","DELAY_DUE_CARRIER","DELAY_DUE_WEATHER","DELAY_DUE_NAS","DELAY_DUE_SECURITY","DELAY_DUE_LATE_AIRCRAFT"]]

#OrdinalEncoder:DEST
le=OrdinalEncoder()
input_colneed["DEST"]=le.fit_transform(input_colneed[["DEST"]])

#(刪除cancelled或diverted資料)在input_colneed中，把欄位DEP_DELAY~AIR_TIME只要有na的那些資料筆數刪掉
input_colneed.dropna(subset=["DEP_DELAY","TAXI_OUT","TAXI_IN","ARR_DELAY","AIR_TIME"],axis=0,how="any",inplace=True)

#在input_colneed中，歸因們是na者用0代替。
input_colneed.fillna(0,inplace=True)

#先看目前自變數是否彼此存在高相關性,考慮建置線性回歸方程式,欄位是否有多重共線性（Multicollinearity）問題
correlation_matrix = input_colneed.corr()  #ANS: 兩兩相關性<0.7，故先不考量多重共線性問題
# 使用 Seaborn 繪製熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()
'''
得知AIR_TIME與距離高度相關，擇一: 捨去飛行距離
'''
input_colneed.drop("AIR_TIME",axis=1,inplace=True)


#%% 個案討論: 針對SFO起飛，UAs的資料: 描述性
# 看DEP_DELAY多久會明顯(?)造成ARR_DELAY?
#input_colneed["DEP_DELAY"]和["ARR_DELAY"]相關性
plt.xlabel("DEP_DELAY")
plt.ylabel("ARR_DELAY")
plt.scatter(input_colneed["DEP_DELAY"],input_colneed["ARR_DELAY"])
delay_r=pd.DataFrame({"DEP_DELAY":input_colneed["DEP_DELAY"],"ARR_DELAY":input_colneed["ARR_DELAY"]})
delay_r.corr()  # DEP_DELAY與ARR_DELAY相關性: 0.959564
plt.show()

# 在DEP_DELAY情形下，造成ARR_DELAY機率? ARR_DELAY>15min機率?
caseDEP_=input_colneed["DEP_DELAY"]>0
caseDEP=input_colneed[caseDEP_]
caseARR_=caseDEP["ARR_DELAY"]>0
caseARR=caseDEP[caseARR_]
caseDEPARR=len(caseARR)/len(caseDEP)
print("在DEP_DELAY情形下，造成ARR_DELAY機率是:%.4f%%"%(caseDEPARR*100))
caseARR_15=caseDEP["ARR_DELAY"]>15
caseARR15=caseDEP[caseARR_15]
caseDEPARR15=len(caseARR15)/len(caseDEP)
print("在DEP_DELAY情形下，造成ARR_DELAY>15min機率是:%.4f%%"%(caseDEPARR15*100))

# 在DEP_DELAY且造成ARR_DELAY情形下，(delay>15分才有歸因)各歸因延誤比例(個別原因數值相加/ARRDEALY>15數值相加)?
def reasonRatio():
    col=caseARR15.columns[6:12]
    for _ in col:
        print(_,":",end="")
        _=caseARR15[_].sum()
        caseDELAY15=caseARR15["ARR_DELAY"].sum()
        print("%.2f%%"%((_/caseDELAY15)*100))
reasonRatio()


#%% 個案討論: 針對SFO起飛，UAs的資料: 描述性
# 線性回歸方程式放入哪些欄位：1.原本，2.三個相關性最高，3.DEP_DELAY -->ANS: 原本都放是最好的(由R2,MSE判斷)
#以下嘗試只有DEP_DELAY,DELAY_DUE_CARRIER,DELAY_DUE_LATE_AIRCRAFT
x=input_colneed.loc[:,["DEP_DELAY","DELAY_DUE_CARRIER","DELAY_DUE_LATE_AIRCRAFT"]]
y=input_colneed.loc[:,"ARR_DELAY"]

lm=LinearRegression()
lm.fit(x,y)
print("ARR_DELAY線性回歸結果：")
coefs=zip(x.columns,lm.coef_)
#for coef in coefs:
    #print(coef)    
print("截距: ",lm.intercept_)
print("準確率: ",lm.score(x,y))
pred=lm.predict(x)
print("MSE: ",mse(y,pred))
print()


#以下嘗試只有DEP_DELAY
x=input_colneed.loc[:,["DEP_DELAY"]]
y=input_colneed.loc[:,"ARR_DELAY"]

lm=LinearRegression()
lm.fit(x,y)
print("ARR_DELAY線性回歸結果：")
coefs=zip(x.columns,lm.coef_)
#for coef in coefs:
    #print(coef)    
print("截距: ",lm.intercept_)
print("準確率: ",lm.score(x,y))
pred=lm.predict(x)
print("MSE: ",mse(y,pred))
print()

#%%執行回歸方程式：
#建立自變數和應變數
x=input_colneed.drop("ARR_DELAY",axis=1)
y=input_colneed.loc[:,"ARR_DELAY"]

#線性回歸: 不分割資料集
lm=LinearRegression()
lm.fit(x,y)
print("ARR_DELAY線性回歸結果：")
coefs=zip(x.columns,lm.coef_)
for coef in coefs:
    print(coef)    
print("截距: ",lm.intercept_.round(4))
print("R-squared: ",lm.score(x,y).round(4))
pred=lm.predict(x)
print("MSE: ",mse(y,pred).round(4))
print()

#線性回歸分割資料: 8:2  
xtrain82,xtest82,ytrain82,ytest82=tts(x,y,test_size=0.2,random_state=100)
lm82=LinearRegression()
lm82.fit(xtrain82,ytrain82)
print("分割資料82 ARR_DELAY線性回歸結果：")
coefs82=zip(xtrain82.columns,lm82.coef_)
for coef82 in coefs82:
    print(coef82)    
print("截距82: ",lm82.intercept_.round(4))
print("R-squared82_train: ",lm82.score(xtrain82,ytrain82).round(4))
print("R-squared82_test: ",lm82.score(xtest82,ytest82).round(4))
pred82=lm82.predict(xtest82)
print("MSE82_test: ",mse(ytest82,pred82).round(4))
plt.scatter(ytest82,pred82)
plt.show()
print()

#線性回歸分割資料: 7.5:2.5
xtrain_,xtest_,ytrain_,ytest_=tts(x,y,test_size=0.25,random_state=100)
lm_=LinearRegression()
lm_.fit(xtrain_,ytrain_)
print("分割資料7.5 2.5 ARR_DELAY線性回歸結果：")
coefs_=zip(xtrain_.columns,lm_.coef_)
for coef_ in coefs_:
    print(coef_)    
print("截距7.5 2.5: ",lm_.intercept_.round(4))
print("R-squared7.5 2.5_train: ",lm_.score(xtrain_,ytrain_).round(4))
print("R-squared7.5 2.5_test: ",lm_.score(xtest_,ytest_).round(4))
pred_=lm_.predict(xtest_)
print("MSE7.5 2.5: ",mse(ytest_,pred_).round(4))
print()

#線性回歸分割資料: 7:3
xtrain73,xtest73,ytrain73,ytest73=tts(x,y,test_size=0.3,random_state=100)
lm73=LinearRegression()
lm73.fit(xtrain73,ytrain73)
print("分割資料73 ARR_DELAY線性回歸結果：")
coefs73=zip(xtrain73.columns,lm73.coef_)
for coef73 in coefs73:
    print(coef73)    
print("截距73: ",lm73.intercept_.round(4))
print("R-squared73_train: ",lm73.score(xtrain73,ytrain73).round(4))
print("R-squared73_test: ",lm73.score(xtest73,ytest73).round(4))
pred73=lm73.predict(xtest73)
print("MSE73: ",mse(ytest73,pred73).round(4))
print()

#線性回歸分割資料: 6:4
xtrain64,xtest64,ytrain64,ytest64=tts(x,y,test_size=0.4,random_state=100)
lm64=LinearRegression()
lm64.fit(xtrain64,ytrain64)
print("分割資料64 ARR_DELAY線性回歸結果：")
coefs64=zip(xtrain64.columns,lm64.coef_)
for coef64 in coefs64:
    print(coef64)    
print("截距64: ",lm64.intercept_.round(4))
print("R-squared64_train: ",lm64.score(xtrain64,ytrain64).round(4))
print("R-squared64_test: ",lm64.score(xtest64,ytest64).round(4))
pred64=lm64.predict(xtest64)
print("MSE64: ",mse(ytest64,pred64).round(4))
print()
#=======END======


#%%把有用到的欄位資料寫入MySQL資料庫
try:
    # 連接 MySQL 資料庫
    con = MySQLdb.connect(
        host="127.0.0.1",
        user="root",
        password="m987654321",
        port=3306,
        database="project_flightdelay"
    )
    
    # 使用 cursor() 做資料庫
    cur = con.cursor()

    # 建立資料表：*僅匯入本專案用到的欄位們
    sql = '''
    CREATE TABLE IF NOT EXISTS flight(
        FL_DATE DATE,
        AIRLINE VARCHAR(100),
        AIRLINE_DOT VARCHAR(100),
        AIRLINE_CODE VARCHAR(100),
        FL_NUMBER SMALLINT,
        ORIGIN VARCHAR(100),
        DEP_DELAY SMALLINT,
        TAXI_OUT SMALLINT,
        TAXI_IN SMALLINT,
        ARR_DELAY SMALLINT,
        CANCELLED TINYINT,
        CANCELLATION_CODE VARCHAR(100),
        DIVERTED TINYINT,
        AIR_TIME INT,
        DISTANCE INT,
        DELAY_DUE_CARRIER SMALLINT,
        DELAY_DUE_WEATHER SMALLINT,
        DELAY_DUE_NAS SMALLINT,
        DELAY_DUE_SECURITY SMALLINT,
        DELAY_DUE_LATE_AIRCRAFT SMALLINT
    )
    '''

    # 執行 SQL 語句
    cur.execute(sql)
    print("資料表建立完畢")

    # 關閉 cursor 和資料庫連接
    cur.close()
    con.close()
    print("table-flight built")
except Exception as e:
    print("Error:", e)

#%% 寫入MySQL
#為了把資料寫入資料庫，把欄位中nan資料以0代替：合理因為1.已做為資料分析與統計，2.被取消或轉降的航班
data22_23_UAs = data22_23_UAs.fillna(0)

try:
    # 開啟資料庫連接
    conn = MySQLdb.connect(host="127.0.0.1",     # 主機名稱
                           user="root",          # 帳號
                           password="m987654321",  # 密碼
                           database = "project_flightdelay",  # 資料庫
                           port=3306,            # port
                           charset="utf8")      # 資料庫編碼
    
    # 使用cursor()方法操作資料庫
    cursor = conn.cursor()
    
    # 將資料data寫到資料庫中
    try:
        for i in range(len(data22_23_UAs)):
            sql = """INSERT INTO flight (FL_DATE,
            AIRLINE,
            AIRLINE_DOT,
            AIRLINE_CODE,
            FL_NUMBER,
            ORIGIN,
            DEP_DELAY,
            TAXI_OUT,
            TAXI_IN,
            ARR_DELAY,
            CANCELLED,
            CANCELLATION_CODE,
            DIVERTED,
            AIR_TIME,
            DISTANCE,
            DELAY_DUE_CARRIER,
            DELAY_DUE_WEATHER,
            DELAY_DUE_NAS,
            DELAY_DUE_SECURITY,
            DELAY_DUE_LATE_AIRCRAFT)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            var = (data22_23_UAs.iloc[i,0], data22_23_UAs.iloc[i,1], data22_23_UAs.iloc[i,2], data22_23_UAs.iloc[i,3],
                   data22_23_UAs.iloc[i,5], data22_23_UAs.iloc[i,6], data22_23_UAs.iloc[i,12], data22_23_UAs.iloc[i,13],
                   data22_23_UAs.iloc[i,16], data22_23_UAs.iloc[i,19], data22_23_UAs.iloc[i,20], data22_23_UAs.iloc[i,21], data22_23_UAs.iloc[i,22],
                   data22_23_UAs.iloc[i,25], data22_23_UAs.iloc[i,26], data22_23_UAs.iloc[i,27], data22_23_UAs.iloc[i,28], data22_23_UAs.iloc[i,29],
                   data22_23_UAs.iloc[i,30], data22_23_UAs.iloc[i,31])     
            
            cursor.execute(sql, var)
            
        conn.commit()        # 提交資料
        
        print("資料寫入完成")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print("錯誤訊息：", e)
 
except Exception as e:
    print("資料庫連接失敗：", e)
    
finally:
    print("資料庫連線結束")

