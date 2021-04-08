
# DSAI-HW2


## run ##


```
conda install --yes --file requirements.txt
```



```
python app.py --training training.csv -- testing testing.csv --output output.csv
```


## train ##

使用GRU做股票預測



將讀取到的training data後29天結合testing data第一天做predict

得到第二天預測出的開盤價後與第一天做比較，進而得出Action

## Action ##

如果預測的隔一天股票開盤大於今天，又已經有股票了就"0"，沒有就"-1"，已經賣空了則"0"

如果預測隔一天的股票開盤小於今天，又已經有股票了就"-1"，沒有就"1"，已經賣空了則"1"

## 20天預測結果 ##

![Fe_r_plot](https://user-images.githubusercontent.com/66662065/114045032-4d1cbc80-98ba-11eb-8ae4-7b82690af664.png)
