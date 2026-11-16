# IoT Data Analysis and Application

「物聯網數據分析與應用」課程作業。兩份作業都依 CRISP-DM 流程進行，並在與 ChatGPT 協作下完成（對話紀錄與 prompt 放在各資料夾的 `note*.txt`）。

## HW1-1：互動式線性迴歸

用 Flask 做一個網頁，透過滑桿即時調整參數，產生隨機資料點並畫出迴歸線。

- 可調參數：斜率 a、截距 b、雜訊係數 c、變異數、資料點數量 n
- 資料點以藍色呈現，迴歸線以紅色呈現
- matplotlib（Agg 後端）繪圖後以 base64 傳到前端

## HW1-2：台積電（2330）股價預測

以多元線性迴歸預測台積電股價，並比較不同特徵組合的效果。

- 資料：`2330-training.csv`，以前幾日收盤價（x1–x5）預測當日股價（y）
- 特徵選擇：Mutual Information、RFE、SelectKBest
- 評估指標：RMSE、R²，並列出所有特徵組合的結果
- 網頁上可自選特徵、查看實際值 vs 預測值圖表

## 技術

Python · Flask · scikit-learn · pandas · NumPy · matplotlib · Bootstrap

## 執行方式

```bash
pip install flask scikit-learn pandas numpy matplotlib joblib
cd HW1_1   # 或 HW1_2
python app.py           # HW1_2 為 python stockPredict.py
```
