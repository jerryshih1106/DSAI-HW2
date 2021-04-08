
# DSAI-HW2


## Run ##


```
conda install --yes --file requirements.txt
```



```
python app.py --training training.csv -- testing testing.csv --output output.csv
```


## Train ##

使用GRU做股票預測



將讀取到的training data後29天結合testing data第一天做predict

得到第二天預測出的開盤價後與第一天做比較，進而得出Action

## Action ##

主要分成以下幾種情況:

##**隔一天預測的股票開盤大於今天**

  
  若已經有股票了，差距大於等於0.5塊則隔天賣掉，小於則持平多觀察陣子
  
  若尚未有股票，差距大於等於0.5塊則隔天賣空，小於則不動作
  
  若尚已經賣空了，則不動作
  
  
  
##**隔一天預測的股票開盤小於今天**


  若已經有股票了，差距大於等於0.5塊則隔天賣掉，小於則持平多觀察陣子
  
  若尚未有股票，差距大於等於0.5塊則隔天買進，小於則不動作
  
  若尚已經賣空了，則買進賺價差
  
  
##**若隔天預測的股票與今日相等，則不動作**


## 20天預測結果 ##

![Fe_r_plot](https://user-images.githubusercontent.com/66662065/114045032-4d1cbc80-98ba-11eb-8ae4-7b82690af664.png)
