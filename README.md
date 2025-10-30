# Simpsons Character Classification Project Report

## I. Experimental Objective | 一、實驗目的
This project aims to build a deep learning model capable of automatically identifying **50 characters** from *The Simpsons*.  
By leveraging the **ResNet101** pre-trained model along with various data augmentation and regularization techniques, the model's classification accuracy under diverse facial expressions and background conditions is significantly improved.  
The final model achieves **91% top-1 accuracy** on the validation set, demonstrating the effectiveness of the applied strategies.  
本專案旨在建立一個能自動辨識《辛普森家庭》(The Simpsons) 中 **50 位角色** 的深度學習模型。  
透過 **ResNet101** 預訓練模型，結合多樣化資料增強與正則化技術，提升模型在多樣臉部表情與背景條件下的辨識準確率。  
最終模型於驗證集上達到 **91% 準確率 (Top-1 Accuracy)**，顯示所採策略能顯著改善分類表現。

---

## II. Dataset and Preprocessing | 二、資料集與前處理

### (1) Dataset Source | (1) 資料來源
- Dataset: *The Simpsons* character images (approx. **97,000 images** in total)  
- Number of Classes: **50 characters**  
- Training/Validation Split: **70%/30%**  
- The class distribution is approximately balanced.  
資料集為《The Simpsons》50 名主要角色影像，總計約 **97,000 張圖片**。  
每個角色平均約 1,800–2,000 張影像，資料分布均衡。

### (2) Data Augmentation Strategy | (2) 資料增強策略
| Type | Technique | Description |  
|------|-----------|-------------|  
| Geometric Transformations | RandomResizedCrop, Rotation, Perspective, ElasticTransform | Simulate different angles and scales |  
| Color Augmentations | ColorJitter, Grayscale, Invert, Solarize | Simulate different lighting and color environments |  
| Background Mixing | BackgroundMixing (p=0.35) | Mix character with random background to reduce background bias |  
| Noise Handling | GaussianBlur, RandomErasing | Simulate noise and occlusion |  
| Combination Techniques | MixUp, CutMix | Combine sample features and labels |  
| Proportions | Weak Augmentation 30%, Strong Augmentation 70% |  
| 類別 | 方法 | 說明 |  
|------|------|------|  
| 幾何變換 | RandomResizedCrop、Rotation、Perspective、ElasticTransform | 模擬多角度與比例差異 |  
| 顏色調整 | ColorJitter、Grayscale、Invert、Solarize | 模擬不同光線與顏色環境 |  
| 背景混合 | BackgroundMixing (p=0.35) | 將角色與隨機背景融合，降低背景偏差 |  
| 雜訊與遮蔽 | GaussianBlur、RandomErasing | 模擬模糊與遮擋 |  
| 資料混合 | MixUp、CutMix | 結合樣本特徵與標籤 |  
| 使用比例 | 原圖弱增強 30% ：強增強 70% |

This strategy significantly improves the model's robustness to **varied backgrounds and facial expressions**.  
此策略可顯著提升模型對 **多樣場景與表情變化** 的魯棒性。

---

## III. Model Architecture and Training Configuration | 三、模型架構與訓練設定

| Item | Setting |  
|------|---------|  
| Model Architecture | ResNet101 (Pre-trained on ImageNet) |  
| Optimizer | AdamW |  
| Loss Function | CrossEntropyLoss (Label Smoothing = 0.03) |  
| Batch Size | 16 |  
| Initial Learning Rate | 0.0002 |  
| Weight Decay | 0.0003 |  
| Epochs | 25 |  
| Learning Rate Strategy | ReduceLROnPlateau |  
| Learning Rate Adjustment | FC layers ×1.0, other layers ×0.1 |  
| Freezing Strategy | Initially freeze all layers, only train `fc`; Unfreeze `layer4` at epoch 5 |  
| Stability Mechanism | EMA (Exponential Moving Average) |  

---

## IV. Training Process | 四、訓練流程
1. Initialize the ResNet101 model and load ImageNet weights.  
2. Freeze all convolutional layers and train only the fully connected (`fc`) layers.  
3. Unfreeze `layer4` at epoch 5 to fine-tune higher-level semantic features.  
4. Apply **differentiated learning rates**:  
 - Higher learning rate for `fc` layers to adapt to the new task.  
 - Lower learning rate for other layers to retain ImageNet features.  
5. Use **MixUp** and **CutMix** to enhance sample diversity.  
6. Monitor accuracy and loss after each epoch using the validation set and save the best weights.  

1. 初始化 ResNet101 模型並載入 ImageNet 權重。  
2. 凍結所有卷積層，只訓練最後的全連接層 (`fc`)。  
3. 第 5 個 epoch 自動解凍 `layer4`，讓模型能進一步微調高階語義特徵。  
4. 透過 **差異化學習率**：  
 - `fc` 層學習率較高，以適應新任務。  
 - 其他層學習率較低，保留通用特徵。  
5. 透過 **MixUp** 與 **CutMix** 提高樣本多樣性。  
6. 每輪訓練後以驗證集監控準確率與 Loss，並自動儲存最佳權重。

---

## V. Experimental Results | 五、實驗結果

### (1) Validation Performance | (1) 驗證集表現
| Metric | Value |  
|--------|-------|  
| Top-1 Accuracy | **0.91** |  
| Loss | 0.25 |  
| Confusion Matrix Size | 50 × 50 |  

The model performs consistently across most characters, with minor misclassifications involving visually similar characters:  
模型在大部分角色的分類上表現穩定，少數誤判集中於外觀相似角色：
- `bart_simpson` ↔ `lisa_simpson`  
- `homer_simpson` ↔ `abraham_grampa_simpson`  
- `nelson_muntz` ↔ `barney_gumble`

---

### (2) Confusion Matrix | (2) 混淆矩陣
The confusion matrix for the model’s predictions on the validation set:  
模型預測結果的混淆矩陣如下圖，橫軸為真實類別、縱軸為預測類別：  

![Confusion Matrix](confusion_matrix.png)

---

### (3) Visualization and Understanding Convolutional Neural Networks | (3) 視覺化與卷積神經網絡
- Successfully visualized the **3x3** and **1x1 convolution kernels** from the final convolutional layer, which helps understand how the model extracts features.  
- Below are the feature maps, which show how the model activates for different regions of the face.

- 成功視覺化了模型的最後一層卷積核（3x3）和1x1卷積核，這有助於理解模型如何提取特徵。  
- 下方還展示了特徵圖，這些特徵圖顯示了模型對不同面部區域的反應，並展示了模型在不同層次如何逐步學習到更高級的特徵。

#### (a) 3x3 Filter Weights (Convolution Kernels) | (a) 3x3 濾波器的權重（卷積核）
![3x3 Conv Kernels](layer4_conv2_kernels_3x3.png)  
This image shows the **3x3 filters** from the final convolutional layer (conv2) of the ResNet model. These filters represent how the model learns to detect local features in the input image, such as edges and color changes.  
這張圖片顯示了 ResNet 模型中最後一層卷積層 (conv2) 的 3x3 濾波器。這些濾波器代表了模型如何學習到輸入圖像中的局部特徵，例如邊緣、顏色變化等。

#### (b) 1x1 Filter Weights (Convolution Kernels) | (b) 1x1 濾波器的權重（卷積核）
![1x1 Conv Kernels](layer4_conv3_kernels_1x1.png)  
This image shows the **1x1 filters** from the final convolutional layer. These filters are typically used for channel-wise semantic mixing, and although their spatial structure is less intuitive than 3x3 filters, they play a crucial role in feature extraction and channel aggregation.  
這張圖片顯示了 ResNet 模型中最後一層 1x1 卷積層的濾波器。這些濾波器通常用於進行通道間的語義混合，雖然它們的空間結構不如 3x3 濾波器直觀，但仍然對特徵提取和通道聚合有著重要作用。

#### (c) Feature Map from the Last Layer | (c) 最後一層的特徵圖（Feature Map）
![Feature Map 1](layer4_feature_map_7x7.png)  
This image shows the **feature map** after the final convolutional layer. Each feature map reflects the model’s activation for certain regions of the image, which may correspond to key features (such as facial parts).  
這張圖顯示了模型在進行前向傳播時，經過最後一層卷積層的特徵圖。每個特徵圖反映了模型對圖像中某些區域的激活，這些區域可能與特徵（如臉部部位）相關。

#### (d) Another Feature Map from the Same Layer | (d) 另一個特徵圖（Feature Map）
![Feature Map 2](layer4_feature_map_7x7_v2.png)  
This image shows another feature map extracted from the same convolutional layer. Similar to the previous one, it visualizes how the model responds to different areas of the image.  
這張圖展示了從同一層卷積中提取的另一個特徵圖，與上一張特徵圖相似，也用來展示模型對不同區域的激活情況。

---

## VI. Performance Comparison and Analysis | 六、效能比較與提升分析

| Model Version | Accuracy | Key Improvements |  
|---------------|----------|------------------|  
| baseline (final.ipynb) | 0.76 | No BackgroundMixing, MixUp |  
| improved (0.91_test.ipynb) | **0.91** | Added BackgroundMixing, MixUp/CutMix, Label Smoothing, EMA, Differentiated LR |  

**Key Reasons for Improvement:**
1. **Background Mixing** reduces background interference, allowing the model to focus more on the character features.  
2. **Differentiated Learning Rates** allow the higher layers to quickly learn the new task, while preserving the general features from ImageNet in the lower layers.  
3. **Label Smoothing** reduces model overconfidence on single classes, improving generalization.  
4. **Strong Augmentation (70%)** increases sample diversity, enhancing the model's robustness.  
5. **EMA** improves the stability and performance of the model during training.

---

## VII. Conclusion and Observations | 七、結論與觀察
- The model performs well in recognizing **The Simpsons** characters, achieving high accuracy even under diverse backgrounds and lighting conditions.  
- The enhanced data augmentation strategies significantly improve the model's generalization capability.  
- The differentiated learning rates and label smoothing prevent overfitting and ensure efficient training.  
- The overall accuracy improved from **0.76** (baseline) to **0.91**, indicating the significant impact of the applied strategies.  

模型能準確辨識《辛普森家庭》主要角色，並在不同背景、光線下維持穩定表現。  
增強策略顯著提升模型的泛化能力。  
差異化學習率使訓練更高效，避免低層特徵被破壞。  
整體準確率由 **0.76**（baseline）提升至 **0.91**，顯示策略顯著改善。

---

## VIII. Future Directions | 八、未來展望
1. Implement **Grad-CAM** or other explainability techniques to visualize the regions the model focuses on.  
2. Explore **semi-supervised learning** to handle unknown characters.  
3. Extend the model’s application to **multi-character detection** or **scene classification** in animated series.  
4. Further optimize the **Background Mixing** strategy to reduce semantic misalignment.

1. 加入 **Grad-CAM** 或其他可解釋性技術，以可視化模型關注區域。  
2. 探索 **半監督學習** 以處理未知角色。  
3. 延伸應用至 **多角色同框場景檢測** 或 **動畫鏡頭分類**。  
4. 進一步優化 **背景混合** 策略以減少語義錯配。

---

## IX. Final Performance | 九、最終效能

> 🎯 **Validation Accuracy:** **91%**  
> 🚀 **Loss:** **0.25**  
> 📈 **Model:** ResNet101 + MixUp + CutMix + LabelSmoothing + BackgroundMixing  

> 🎯 **驗證準確率：** **91%**  
> 🚀 **Loss：** **0.25**  
> 📈 **模型：** ResNet101 + MixUp + CutMix + LabelSmoothing + BackgroundMixing  

---

## X. Development Environment and Versions | 十、開發環境與版本

| Package | Version |  
|---------|---------|  
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

This concludes the report for the **Simpsons Character Classification** project.  
這是《辛普森家庭角色分類》專案的報告結尾。

---
