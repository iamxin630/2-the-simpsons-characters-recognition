# Simpsons 角色分類專案報告

## 一、實驗目的
本專案旨在建立一個能自動辨識《辛普森家庭》(The Simpsons) 中 50 個角色的深度學習模型。  
透過 ResNet101 預訓練模型，結合多樣化資料增強與正則化技術，提升模型在多樣臉部表情與背景條件下的辨識準確率。

---

## 二、資料與前處理

### (1) 資料集
- 來源：Simpsons 角色影像集（共約 97,000 張圖片）
- 分類數：50 個角色
- 訓練/驗證比：7:3
- 每類樣本數約略平衡

### (2) 資料增強策略
| 類型 | 技術說明 |
|------|------------|
| 幾何變換 | RandomResizedCrop、Rotation、Perspective、ElasticTransform |
| 顏色增強 | ColorJitter、Grayscale、Invert、Solarize |
| 背景替換 | BackgroundMixing (p=0.35) |
| 雜訊處理 | GaussianBlur、RandomErasing |
| 組合技術 | MixUp、CutMix |
| 比例設定 | 原圖弱增強 30%、強增強 70% |

---

## 三、模型架構與訓練設定

| 項目 | 設定 |
|------|------|
| 模型架構 | ResNet101 (ImageNet 預訓練) |
| 最後輸出 | 全連接層輸出 50 維 Softmax |
| 優化器 | AdamW |
| 初始學習率 | 0.0002 |
| 差異化學習率 | FC層 ×1.0, 其他層 ×0.1 |
| 損失函數 | CrossEntropyLoss (Label Smoothing=0.03) |
| 權重衰減 | 0.0003 |
| 批次大小 | 16 |
| Epochs | 25 |
| 學習率調整 | ReduceLROnPlateau |
| 模型穩定化 | EMA (Exponential Moving Average) |

---

## 四、實驗結果

### 準確率表現
| 指標 | 數值 |
|------|------|
| Training Accuracy | 0.94 |
| Validation Accuracy | 0.91 |
| Validation Loss | 0.25 |

### Task 1: 預測 50 個辛普森角色
- 使用 ResNet101 模型預測圖片中的辛普森角色，成功辨識了所有50個角色的類別。模型在多樣的面部表情和背景變化下依然保持了良好的分類效果。

### Task 2: 計算混淆矩陣
- 使用50x50的混淆矩陣，觀察每一類別的預測結果。混淆矩陣顯示出大多數角色的辨識精度很高，但仍有些類別會被誤分類。
![Confusion Matrix](confusion_matrix.png)

### Task 3: 視覺化與理解卷積神經網絡
- 我們成功可視化了模型的最後一層卷積核（3x3）和1x1卷積核，這有助於我們理解模型如何提取臉部特徵。

- 我們還展示了特徵圖，這些特徵圖顯示了模型對不同面部區域的反應，並展示了模型在不同層次如何逐步學習到更高級的特徵。

---

## 五、結論與觀察

1. 透過強化資料增強 (尤其 BackgroundMixing + MixUp)，模型更具泛化性。  
2. 差異化學習率設定使高層能快速學習新任務，而底層保留 ImageNet 特徵。  
3. Label Smoothing 有效避免模型對單一類別過度自信。  
4. 整體準確率由 baseline (0.76) 提升至 (0.91)，顯示策略顯著改善。  

---

## 六、未來展望
- 導入 Grad-CAM 進一步分析模型關注區域。
- 加入半監督學習策略處理未知角色。
- 延伸應用至多角色場景之檢測任務。

---

📌 **最終成績**  
> Validation Accuracy：**91%**  
> Loss：**0.25**  
> 模型：ResNet101 + MixUp + CutMix + LabelSmoothing + BackgroundMixing
