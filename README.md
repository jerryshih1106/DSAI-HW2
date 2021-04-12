
# DSAI-HW2


## Run ##


## 安裝 ##
```
pip install -r requirements.txt
```

## 執行 ##
```
python app.py --training training.csv --testing testing.csv --output output.csv
```


## Train ##

使用GRU做股票預測

將training data資料作訓練後

將讀取到的training data後29天結合testing data第一天做predict

得到第2天預測出的開盤價後與第一天正解做比較，進而得出第2天的動作

隔天得到第二天的正解後，將讀取到的training data後的28天結合testing data的第1,2天predict第3天的開盤價

以此類推。

## Action ##

主要分成以下幾種情況:

### **隔一天預測的股票開盤大於今天** <h3>

  若已經有股票了，差距大於等於gap(預設0.5元)則隔天賣掉，小於則持平多觀察陣子
  
  若尚未有股票，差距大於等於gap則隔天賣空，小於則不動作
  
  若已經賣空了，則不動作(賣了賠錢)
  
### **隔一天預測的股票開盤小於今天** <h3>

  若已經有股票了，差距大於等於gap則隔天賣掉，小於則持平多觀察陣子
  
  若尚未有股票，差距大於等於gap則隔天買進，小於則不動作
  
  若已經賣空了，則先判斷說明天的價格是否比賣空的價格低(避免依然賠錢的情況)
    
### **若隔天預測的股票與今日相等，則不動作** <h3> 
```
if 預測明天開盤>今天開盤:
  if 有股票:
    if 兩天差距 >= gap:
      -1
    if 兩天差距 < gap:
      1
  if 沒有股票:
    if 兩天差距 >= gap:
      -1
    if 兩天差距 < gap:
      0
  if 賣空股票:
    0
if 預測明天開盤<今天開盤:
  if 有股票:
    if 兩天差距 >= gap:
      -1
    if 兩天差距 < gap:
      0
  if 沒有股票:
    if 兩天差距 >= gap:
      1
    if 兩天差距 < gap:
      0
  if 賣空股票:
    if 賣空價格 <= 明天預測開盤:
      0
    if 賣空價格 > 明天預測開盤:
      1
 if 兩天相等:
    0
```
## 20天預測結果 ##

![Fe_r_plot](https://user-images.githubusercontent.com/66662065/114268892-1d92bf00-9a36-11eb-9109-6756aa4c0409.png)

## EG Action ##


```
Action
1
-1
1
0
0
-1
1
-1
.
.
.

```
