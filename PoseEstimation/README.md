
# カメラ位置毎の学習に基づく人物姿勢推定[[1]]
## 目次

- 事前に用意するもの
- 使い方
- 論文中の実験再現
- ディレクトリ構造

## 事前に用意するもの

- ライブラリ
    - scikit-learn
    - Numpy
    - Scipy
    - Pandas
    - Cython
    - Pillow
    - primesense
    - opencv3

    ※ opencv3以外はpipでインストール可能
- ソフトウェア
    - Poser Pro 2014[[3]]
- データセット
    - CMU Mocapデータセット[[4]]

## 使い方
### 概要

1. Poserによるデータ生成
2. 実データの取得
3. 従来手法[[2]]の実行
4. 提案手法[[1]]の実行(カメラ位置の離散化)
5. 提案手法の実行(人物姿勢推定)
6. 合成データに対する精度評価
7. 実データに対する引き出し推定と精度評価

### 1. Poserによるデータ生成
#### 実行方法
1. Poserを開く
2. ファイル > スクリプトの実行 で，Script/Preprocessing/Poser/make\_(train|test).py を選択

※ 基本的には0.5sec程度で1組のデータが生成されるが，1日程経つと生成速度がかなり落ちるため，一旦Poserを強制終了した後に，実行し直すという手順を取ったほうが生成が早く終わる．

#### 入力
- 姿勢データ : 
    - Data/Preprocessing/MotionBVH/Regularized/\*/\*.bvh
- Poserデータ : 
    - Data/Preprocessing/Poser/(female|male).pz3

#### 出力
- 合成学習データ(深度画像，部位ラベル，Poser内パラメタの組 15,000×64組) : 
    - Data/Main/BodyPartClassification/SyntheticImages/train/\*( Z.png|.png|\_param)
- 合成テストデータ(深度画像，部位ラベル，Poser内パラメタの組 100×64組)  : 
    - Data/Main/BodyPartClassification/SyntheticImages/test/\*( Z.png|.png|\_param)


### 2. 実データの取得
#### 実行方法
1. Xtion Pro Live[[5]]にUSBを繋ぐ．
2. カレントディレクトリを移動．
```sh
cd Script/Preprocessing/RealData/
```
3. ipython上で以下を実行 ※TargetPath, videonameは任意
```python
from xtion_io import XtionIO
x = XtionIO()
x.initialize()
x.capture_bg_depth_img($TargetPath+"bg.png") # 背景深度画像の取得
x.capture_depth_video($TargetPath+"$videoname_raw.avi") # 推定対象深度動画の取得
x.release()
```
4. 以下を実行．
```sh
python video_segmentation.py -t TargetPath  # 人物が引き出しに近いフレームのみを抽出
```

#### 入力
- Xtion Pro Liveのカメラ入力

#### 出力
- 背景深度画像と推定対象深度動画とセグメンテーション後の深度画像列
    - Data/Main/BodyPartClassification/CapturedVieos/"$TargetPath"bg.png
    - Data/Main/BodyPartClassification/CapturedVieos/"$TargetPath"$videoname\_raw.avi
    - Data/Main/BodyPartClassification/CapturedImages/"$TargetPath"$videoname\_\*.png

### 2. 従来手法の実行
#### 実行方法

```sh
cd Script/Main/
python body_part_classification.py -t TestPath -n NTrain -N NTest
python joint_position_prediction.py -t TestPath -n NTrain -N NTest
```
TestPathには，合成データに対して実行する時はSyntheticImages/\*male/などと指定．
実データに対して実行する時はCapturedImages/person1/等と指定．

#### 入力
- 合成学習データ(深度画像，部位ラベル，Poser内パラメタの組 $NTrain組)
- テスト深度画像$NTest枚
    - Data/Main/BodyPartClassification/$TestPath\*.png

#### 出力
- 人物姿勢推定結果(関節の3次元位置，描画した関節の2次元位置の組 $NTest組) : 
    - Data/Main/JointPositionPrediction/Output/$TestPath\*\_$NTrain\_JPP(.ply|.png)


### 3. 提案手法の実行(カメラ位置の離散化)
#### 実行方法
```sh
cd Script/Main/
python camera_location_clustering.py -n NTrain -N NTest
```

#### 入力
- 合成学習データ(深度画像，部位ラベル，Poser内パラメタの組 $NTrain×64組)
- 合成テスト深度画像$NTest×64枚

#### 出力
- 離散化設定ファイル : 
    - Data/Main/BodyPartClassification/Intermediate/discr\_setting\_type/type\_$NTrain\_$NTest\_\*.csv

### 4. 提案手法の実行(人物姿勢推定)
#### 実行方法

```sh
cd Script/Main/
python divide_and_conquer_BPC.py -t TestPath -n NTrain -N NTest -D DiscrType
python joint_position_prediction.py -t TestPath -n NTrain -N NTest -D DiscrType
```
TestPathには，合成データに対して実行する時はSyntheticImages/\*male/などと指定．
実データに対して実行する時はCapturedImages/person1/等と指定．

#### 入力
- 合成学習データ(深度画像，部位ラベル，Poser内パラメタの組 $NTrain×64組)
- テスト深度画像$NTest枚
    - Data/Main/BodyPartClassification/$TestPath\*.png
- 離散化設定ファイル

#### 出力
- 人物姿勢推定結果(関節の3次元位置，描画した関節の2次元位置の組 $NTest組) : 
    - Data/Main/JointPositionPrediction/Output/$TestPath\*\_$NTrain\_$DiscrType\_JPP(.ply|.png)


### 5. 合成データに対する精度評価
#### 実行方法
```sh
cd Script/Main/
python JPP_precision.py -t TestPath -n NTrain -N NTest [-D DiscrType]
```
※ DiscrTypeを指定しない場合は従来手法の精度評価．指定した場合は提案手法の精度評価

#### 入力
- 合成データに対する人物姿勢推定結果(関節の3次元位置データ$NTest個)
    - 2.または3.で合成データを$TestPathとして指定することにより取得

#### 出力
- 関節位置推定精度(Average Precision) : 
    - Data/Main/JointPositionPrediction/Evaluation/$TestPath\*\_$NTrain\_$DiscrType.csv

### 6. 実データに対する引き出し推定
#### 実行方法
```sh
cd Script/Main/
python drawer_prediction.py -t TestPath -n NTrain [-D DiscrType]
```
※ DiscrTypeを指定しない場合は従来手法の精度評価．指定した場合は提案手法の精度評価

#### 入力
- 実データに対する人物姿勢推定結果(関節の3次元位置データ$NTest個)
    - 2.または3.で実データを$TestPathとして指定することにより取得

#### 出力
- 深度画像列に対する引き出しの推定結果
    - Data/Main/JointPositionPrediction/Output/$TestPath\*\_drawer.csv
- 深度画像列内の深度画像それぞれに対する引き出しの推定結果
    - Data/Main/JointPositionPrediction/Output/$TestPath\*\_drawers.csv

※ 今回対象とした深度画像列は99列と少なかったため，精度評価の自動化は行わなかった．

## 論文中の実験再現
Poserを用いたデータの生成および実データの取得は事前に終了しているものとする．
### 従来手法との比較
```sh
cd Script/Main/
python camera_location_clustering.py -n 1000 -N 100
python body_part_classification.py -t SyntheticImages/*male/ -n 15000 -N 100
python joint_position_prediction.py -t SyntheticImages/*male/ -n 15000 -N 100
python JPP_precision.py -t SyntheticImages/*male/ -n 15000 -N 100 
python divide_and_conquer_BPC.py -t SyntheticImages/*male/ -n 15000 -N 100 -D type_1000_100_22
python joint_position_prediction.py -t SyntheticImages/*male/ -n 15000 -N 100 -D type_1000_100_22
python JPP_precision.py -t SyntheticImages/*male/ -n 15000 -N 100 -D type_1000_100_22
```

### 離散化方法の最適化
```sh
cd Script/Main/
python camera_location_clustering.py -n 1000 -N 100
python divide_and_conquer_BPC.py -D type_1000_100_24 -n 15000 -N 100
python joint_position_prediction.py -D type_1000_100_24 -n 15000 -N 100
python JPP_precision.py -t SyntheticImages/*male/ -n 15000 -N 100 -D type_1000_100_24
python divide_and_conquer_BPC.py -D type_1000_100_23 -n 15000 -N 100
python joint_position_prediction.py -D type_1000_100_23 -n 15000 -N 100
python JPP_precision.py -t SyntheticImages/*male/ -n 15000 -N 100 -D type_1000_100_23
python divide_and_conquer_BPC.py -D type_1000_100_22 -n 15000 -N 100
python joint_position_prediction.py -D type_1000_100_22 -n 15000 -N 100
python JPP_precision.py -t SyntheticImages/*male/ -n 15000 -N 100 -D type_1000_100_22
```

### 実データへの適用可能性
```sh
cd Script/Main/
python camera_location_clustering.py -n 1000 -N 100
python divide_and_conquer_BPC.py -t CapturedImages/person*/ -n 15000 -N 100000 -D type_1000_100_22
python joint_position_prediction.py -t CapturedImages/person*/ -n 15000 -N 100000 -D type_1000_100_22
python drawer_prediction.py -t CapturedVideos/person*/ -n 15000 -D type_1000_100_22
```

## ディレクトリ構成

- PoseEstimation
    - Script
        - Main
            - body_part_classification.py : 部位分類(従来手法[[2]])
            - divide_and_conquer_BPC.py : 部位分類(提案手法[[1]])
            - joint_position_prediction.py : 関節位置推定(従来手法，提案手法共通)
            - camera_location_clustering.py : 提案手法(カメラ位置離散化)
            - JPP_precision.py : 関節位置推定精度評価
            - drawer_prediction.py : 引き出し推定
            - Modules : モジュール群
            - ...
        - Preprocessing
            - Poser : Poser用データ生成スクリプト群
                - make_train.py
                - make_test.py
            - MotionBVH : BVHファイルから姿勢を減らすのに用いたスクリプト群
            - RealData
                - video_segmentation.py : 動画を画像列に変換(引き出しに近い所のみ抽出)
                - xtion_io.py
                - ... : その他引き出しの正解位置を求めるために用いたスクリプト群

    - Data
        - Main
            - BodyPartClassification
                - SyntheticImages
                    - female
                    - male
                - CapturedImages
                    - person1
                        - $videoname\_\*.png
                    - person2
                    - person3
                - CapturedVideos
                    - person1
                        - bg.png
                        - $videoname\_raw.avi
                    - person2
                    - person3
                - Evaluation
                - GroundTruth
                - Intermediate
                    - pkl : Random Forestをpickle化したものを保存
                    - discretization_setting : 離散化方法を保存
                    - similarity_matrix : 離散化の過程の弱推定器間類似度を保存
                    - ... : その他設定ファイルと特徴量等を保存
                - Output : 出力結果
            - JointPositionPrediction
                - Output : 出力結果
                - Evaluation : 各種評価
                - GroundTruth : 人物姿勢と引き出し位置の正解
                - Byproduct : 副次的に得られる深度画像のPointCloud
        - Preprocessing
            - Poser
                - female.pz3 : Poserデータ
                - male.pz3 : Poserデータ
                - Figures : 調整したフィギュア群
            - MotionBVH
                - labeled_female.bvh : Poser Pro 2014[[4]]内femaleフィギュアの各部位の大きさの参考とするファイル
                - labeled_male.bvh : Poser Pro 2014内のmaleフィギュアの各部位の大きさの参考とするフィアル
                - Raw : CMU Mocap[[3]]から取得したBVHデータ
                - Intermediate : 中間ファイル
                - Regularized : Rawの姿勢を常に前向きにし，どの姿勢も同じ関節同士が互いに5cm以上離れるようデータを間引いた後のBVHデータ

[1]: http://www.mm.media.kyoto-u.ac.jp/wp-content/uploads/2017/03/2016-m-takagi.pdf
[2]: http://sistemas-humano-computacionais.wdfiles.com/local--files/capitulo:modelagem-e-simulacao-de-humanos/BodyPartRecognition%20MSR%2011.pdf
[3]: http://www.poser.jp/products/pro2014/index.html
[4]: http://mocap.cs.cmu.edu/
[5]: https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE/

## 筆者連絡先
質問等あれば高木 和久まで．

E-mail: raghckp92bsk at gmail.com

