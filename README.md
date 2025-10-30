# Simpsons 角色分類專案報告

## 一、實驗目的
本專案旨在建立一個能自動辨識《辛普森家庭》(The Simpsons) 中 **50 位角色** 的深度學習模型。  
透過 **ResNet101** 預訓練模型，結合多樣化資料增強與正則化技術，提升模型在多樣臉部表情與背景條件下的辨識準確率。  
最終模型於驗證集上達到 **91% 準確率 (Top-1 Accuracy)**，顯示所採策略能顯著改善分類表現。

---

## 二、資料集與前處理

### (1) 資料來源
- 資料集為《The Simpsons》50 名主要角色影像，總計約 **97,000 張圖片**。
- 每個角色平均約 1,800–2,000 張影像，資料分布均衡。
- 資料結構如下：

### (2) 資料分割比例
- 訓練集：70%
- 驗證集：30%
- 各類別採 **分層抽樣 (stratified split)**，確保每類樣本比例一致。

### (3) 資料增強策略
| 類別 | 方法 | 說明 |
|------|------|------|
| 幾何變換 | RandomResizedCrop, Rotation, Perspective, ElasticTransform | 模擬多角度與比例差異 |
| 顏色調整 | ColorJitter, Grayscale, Solarize, Invert | 模擬不同光線與顏色環境 |
| 背景混合 | BackgroundMixing (p=0.35) | 將角色與隨機背景融合，降低背景偏差 |
| 雜訊與遮蔽 | GaussianBlur, RandomErasing | 模擬模糊與遮擋 |
| 資料混合 | MixUp, CutMix | 結合樣本特徵與標籤 |
| 使用比例 | 原圖弱增強 30% ：強增強 70% |

此策略可顯著提升模型對 **多樣場景與表情變化** 的魯棒性。

---

## 三、模型架構與訓練設定

| 項目 | 設定 |
|------|------|
| 模型架構 | ResNet101 (ImageNet 預訓練) |
| 優化器 | AdamW |
| 損失函數 | CrossEntropyLoss (Label Smoothing = 0.03) |
| 批次大小 | 16 |
| 初始學習率 | 0.0002 |
| 權重衰減 | 0.0003 |
| 訓練輪數 | 25 |
| 學習率策略 | ReduceLROnPlateau |
| 學習率層級設定 | FC層 ×1.0, 其餘層 ×0.1 |
| 凍結策略 | 初期凍結所有層，僅訓練 `fc`；第5輪解凍 `layer4` |
| 穩定化機制 | EMA (Exponential Moving Average) |

---

## 四、訓練流程
1. 初始化 ResNet101 模型並載入 ImageNet 權重。  
2. 凍結所有卷積層，只訓練最後的全連接層 (`fc`)。  
3. 第 5 個 epoch 自動解凍 `layer4`，讓模型能進一步微調高階語義特徵。  
4. 採用 **差異化學習率**：  
 - `fc` 層學習率較高，以適應新任務。  
 - 其他層學習率較低，保留通用特徵。  
5. 透過 MixUp 與 CutMix 提高樣本多樣性。  
6. 每輪訓練後以驗證集監控準確率與 Loss，並自動儲存最佳權重。  

---

## 五、實驗結果

### (1) 驗證集表現
| 指標 | 數值 |
|------|------|
| Top-1 Accuracy | **0.91** |
| Loss | 0.25 |
| 混淆矩陣維度 | 50 × 50 |

模型在大部分角色的分類上表現穩定，少數誤判集中於外觀相似角色：
- `bart_simpson` ↔ `lisa_simpson`
- `homer_simpson` ↔ `abraham_grampa_simpson`

---

### (2) Confusion Matrix
模型預測結果的混淆矩陣如下圖，橫軸為真實類別、縱軸為預測類別：  

![Confusion Matrix](docs/confmat.png)

---

### (3) 卷積核 (Filter) 視覺化
從 ResNet101 最後一層卷積 (`layer4[-1].conv2`) 可觀察到：
- 濾波器對角色臉部的 **邊緣、輪廓、眼睛與嘴巴區域** 特別敏感。  

![Filter Visualization](docs/filters.png)

---

### (4) 特徵圖 (Feature Map) 視覺化
最後一層卷積特徵圖顯示模型在角色的臉部中心區域具有最強激活值，  
證明模型確實學會關注關鍵辨識特徵（眼睛、嘴型、頭髮輪廓等）。

![Feature Map](docs/featuremaps.png)

---

## 六、效能比較與提升分析

| 模型版本 | 準確率 | 改進重點 |
|-----------|----------|-----------|
| baseline (final.ipynb) | 0.76 | 無 BackgroundMixing、MixUp |
| improved (0.91_test.ipynb) | **0.91** | 加入 BackgroundMixing、MixUp/CutMix、Label Smoothing、EMA、差異化 LR |

**主要提升原因：**
1. **背景混合 (BackgroundMixing)** 降低背景干擾，使模型專注角色特徵。  
2. **差異化學習率** 讓高層快速學習新任務、底層維持穩定。  
3. **Label Smoothing** 減少模型過度自信，提升泛化。  
4. **強增強比例 70%** 增加資料多樣性。  
5. **EMA 平滑更新** 改善訓練穩定性。

---

## 七、結論與觀察
- 模型能準確辨識《辛普森家庭》主要角色，並在不同背景、光線下維持穩定表現。  
- 增強策略顯著提升模型的泛化能力。  
- MixUp 與 CutMix 的組合能有效防止過擬合。  
- 差異化學習率使訓練更高效，避免低層特徵被破壞。  

最終結果：
> 🎯 **Validation Accuracy：91%**  
> 🚀 **Loss：0.25**

---

## 八、未來展望
1. 加入 **Grad-CAM** 等可解釋性分析以可視化模型關注區域。  
2. 探索 **半監督學習** 以擴充資料來源。  
3. 延伸應用至 **多角色同框場景檢測** 或 **動畫鏡頭分類**。  
4. 進一步優化背景混合策略以減少語義錯配。

---

## 九、開發環境與版本

| 套件 | 版本 |
|------|------|
| Python | 3.10 |
| torch | 2.1.2 |
| torchvision | 0.16.2 |
| numpy | 1.26.4 |
| pandas | 2.2.1 |
| scikit-learn | 1.5.1 |
| matplotlib | 3.8.4 |
| Pillow | 10.2.0 |
| tqdm | 4.66.2 |
| opencv-python | 4.9.0.80 |

---



---
