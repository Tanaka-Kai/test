{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##　➁フィルム断裂予測プログラム"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 必要なライブラリーの導入"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd#pandasを開く\n",
    "import numpy as np#numpyを開く\n",
    "import matplotlib.pyplot as plt\n",
    "import japanize_matplotlib \n",
    "import sklearn\n",
    "\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r'C:\\WPy64-31180\\notebooks\\nichicon_data.csv',encoding=\"cp932\")#日本語表示できるようにデータを開く"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### データの合格数と不合格数を確認する"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "合格数  102 個\n",
      "不合格数 21 個\n",
      "サンプル数 123 個\n"
     ]
    }
   ],
   "source": [
    "countg = df['合否'].str.count('合格').sum()\n",
    "countf = df['合否'].str.count('不合格').sum()\n",
    "print('合格数 ',countg - countf ,\"個\") # 出力結果: 2\n",
    "print('不合格数',countf,\"個\") # 出力結果: 2\n",
    "print('サンプル数',countg,\"個\") "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "scrolled": true
   },
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df.iloc[:, 6].map({'不合格': 1, '合格': 0})#合否の値を目的変数に設定する　不合格をクラス1, 合格をクラス0とする\n",
    "x = df.iloc[:, 0:6]#説明変数を設定する。要素番号0から要素番号6のひとつ手前まで\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### データの前処理"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     マージンオイル補充量  マージンオイル交換量  フレキソオイル補充量  フレキソオイル交換量  マージンノズル設定温度  フレキソ設定温度  \\\n",
      "0            30           0          50           0          135       139   \n",
      "1            20           0          58           0          136       140   \n",
      "2            20           0          50           0          136       142   \n",
      "3            30           0          76           0          135       141   \n",
      "4            20           0          50           0          137       144   \n",
      "..          ...         ...         ...         ...          ...       ...   \n",
      "118          30           0         200           0          134       135   \n",
      "119          30           0         200           0          134       135   \n",
      "120          30           0         200           0          134       135   \n",
      "121          30           0         220           0          134       136   \n",
      "122          30           0         120           0          135       136   \n",
      "\n",
      "      合否  \n",
      "0     合格  \n",
      "1     合格  \n",
      "2     合格  \n",
      "3     合格  \n",
      "4     合格  \n",
      "..   ...  \n",
      "118   合格  \n",
      "119   合格  \n",
      "120   合格  \n",
      "121  不合格  \n",
      "122   合格  \n",
      "\n",
      "[123 rows x 7 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "\n",
    "# CSVファイルを読み込む\n",
    "df = pd.read_csv(r'C:\\WPy64-31180\\notebooks\\nichicon_data.csv',encoding=\"cp932\")\n",
    "\n",
    "# 処理する列のリスト\n",
    "columns_to_process = ['マージンオイル補充量', 'マージンオイル交換量', 'フレキソオイル補充量', 'フレキソオイル交換量','マージンノズル設定温度','フレキソ設定温度']\n",
    "\n",
    "# セルの値を処理する関数\n",
    "def process_cell(cell_value):\n",
    "    # 文字列から数字のみを抽出してリストとして返す\n",
    "    numbers = re.findall(r'\\d+', str(cell_value))\n",
    "    if len(numbers) >= 2:\n",
    "        # 数値を整数に変換\n",
    "        num1 = int(numbers[0])\n",
    "        num2 = int(numbers[1])\n",
    "        \n",
    "        # 2番目の数値を4/10倍して加算\n",
    "        new_num2 = int(num2 * 4 / 10)\n",
    "        \n",
    "        # 2つの数値を加算して結果を返す\n",
    "        result = num1 + new_num2\n",
    "        return result\n",
    "    elif numbers:  # 数字が1つの場合\n",
    "        # 数字のみを返す\n",
    "        return int(numbers[0])\n",
    "    else:  # 数字がない場合\n",
    "        return ''\n",
    "\n",
    "# 各列に対して処理を適用\n",
    "for column in columns_to_process:\n",
    "    df[column] = df[column].apply(process_cell)\n",
    "\n",
    "# 処理されたデータを表示する\n",
    "print(df)\n",
    "\n",
    "# 処理されたデータをCSVファイルに書き込む\n",
    "df.to_csv(r'C:\\WPy64-31180\\notebooks\\nichicon_data.csv', encoding=\"cp932\", index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "合格数  102 個\n",
      "不合格数 21 個\n",
      "サンプル数 123 個\n"
     ]
    }
   ],
   "source": [
    "countg = df['合否'].str.count('合格').sum()\n",
    "countf = df['合否'].str.count('不合格').sum()\n",
    "print('合格数 ',countg - countf ,\"個\") # 出力結果: 2\n",
    "print('不合格数',countf,\"個\") # 出力結果: 2\n",
    "print('サンプル数',countg,\"個\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 重みづけ確認"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(21, 102)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(y[y==1]), len(y[y==0])  #不合格(1)、合格(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{1: 5.857142857142857, 0: 1.2058823529411764}"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "weight = {\n",
    "    1:len(y) / len(y[y==1]),\n",
    "    0:len(y)/ len(y[y==0])\n",
    "}\n",
    "weight"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 分類問題"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### モデル作成（XGboost）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,\n",
       "              colsample_bylevel=None, colsample_bynode=None,\n",
       "              colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "              enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "              gamma=None, grow_policy=None, importance_type=None,\n",
       "              interaction_constraints=None, learning_rate=None, max_bin=None,\n",
       "              max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "              max_delta_step=None, max_depth=None, max_leaves=None,\n",
       "              min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "              multi_strategy=None, n_estimators=None, n_jobs=None,\n",
       "              num_parallel_tree=None, random_state=0, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;XGBClassifier<span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,\n",
       "              colsample_bylevel=None, colsample_bynode=None,\n",
       "              colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "              enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "              gamma=None, grow_policy=None, importance_type=None,\n",
       "              interaction_constraints=None, learning_rate=None, max_bin=None,\n",
       "              max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "              max_delta_step=None, max_depth=None, max_leaves=None,\n",
       "              min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "              multi_strategy=None, n_estimators=None, n_jobs=None,\n",
       "              num_parallel_tree=None, random_state=0, ...)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "XGBClassifier(base_score=None, booster=None, callbacks=None,\n",
       "              colsample_bylevel=None, colsample_bynode=None,\n",
       "              colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "              enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "              gamma=None, grow_policy=None, importance_type=None,\n",
       "              interaction_constraints=None, learning_rate=None, max_bin=None,\n",
       "              max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "              max_delta_step=None, max_depth=None, max_leaves=None,\n",
       "              min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "              multi_strategy=None, n_estimators=None, n_jobs=None,\n",
       "              num_parallel_tree=None, random_state=0, ...)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import xgboost as xgb\n",
    "# 各クラスのサンプル数を取得して比率を計算\n",
    "from collections import Counter\n",
    "counter = Counter(y)\n",
    "scale_pos_weight = counter[0] / counter[1]    #クラス1（不合格）の重み増し\n",
    "\n",
    "# xgboostモデルの作成  scale_pos_weightは不均衡なクラスの比率を指定するために使われます。正のクラスの重みを設定することができます。\n",
    "model = xgb.XGBClassifier(random_state=0, scale_pos_weight=scale_pos_weight)\n",
    "\n",
    "# 学習\n",
    "model.fit(x,y)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### モデル評価（クロスバリテーションにて確認）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "各分割におけるAccuracyスコア:\n",
      "分割 1: 0.76\n",
      "分割 2: 0.44\n",
      "分割 3: 0.80\n",
      "分割 4: 0.75\n",
      "分割 5: 0.83\n",
      "\n",
      "統計情報:\n",
      "平均 Accuracy: 0.72\n",
      "標準偏差: 0.14\n",
      "最小 Accuracy: 0.44\n",
      "最大 Accuracy: 0.83\n",
      "適合率: 0.84\n",
      "再現率: 1.00\n",
      "F値: 0.91\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "# XGBoostモデルのインスタンス化などが前提です。\n",
    "\n",
    "# 正確性のクロスバリデーションスコアの計算\n",
    "accuracy_scores = cross_val_score(model, x, y, cv=5, scoring='accuracy')\n",
    "\n",
    "# 各分割におけるスコアの詳細を表示\n",
    "print(\"各分割におけるAccuracyスコア:\")\n",
    "for i, score in enumerate(accuracy_scores, start=1):\n",
    "    print(f\"分割 {i}: {score:.2f}\")\n",
    "\n",
    "# 統計情報の出力\n",
    "print(\"\\n統計情報:\")\n",
    "print(f\"平均 Accuracy: {accuracy_scores.mean():.2f}\")\n",
    "print(f\"標準偏差: {accuracy_scores.std():.2f}\")\n",
    "print(f\"最小 Accuracy: {accuracy_scores.min():.2f}\")\n",
    "print(f\"最大 Accuracy: {accuracy_scores.max():.2f}\")\n",
    "\n",
    "from sklearn.metrics import precision_score, recall_score, f1_score\n",
    "\n",
    "y_pred = model.predict(x)\n",
    "\n",
    "# 適合率を計算\n",
    "precision = precision_score(y, y_pred)\n",
    "\n",
    "# 再現率を計算\n",
    "recall = recall_score(y, y_pred)\n",
    "\n",
    "# F値を計算\n",
    "f1 = f1_score(y, y_pred)\n",
    "\n",
    "print(f\"適合率: {precision:.2f}\")\n",
    "print(f\"再現率: {recall:.2f}\")\n",
    "print(f\"F値: {f1:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAqkAAAIgCAYAAACmmfDXAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAsnUlEQVR4nO3debyf853//+eJRCQhQZHIIpsgRico2i+dpqjaaxlbF0tK1TYkUSojJNWOtBoRpf1haCIoM0yG2kbHNkhsrUFUijR7SBNFgsh+/f7ozRlHEs6JE+ctud9vt9xuPtd1fa7P65zG8ei1nZqqqqoAAEBBmjX1AAAA8GEiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFeAjzJ07N0cccUQ23njjdOvWLbfeemtTjwSwThCpQPGmT5+e0047Ld26dUvLli3TsWPHHHbYYXn00UfX+GefeuqpmThxYu65555ceuml6dOnT6Psd+jQoampqcnUqVMbZX/18fDDD6empibrr79+5syZs8rtdthhh9TU1GTo0KEN/oxnn302f/jDH+q17ejRo1NTU5OHH364wZ8DrP1EKlC0cePGpU+fPhk/fnx++tOf5qmnnsqoUaNSU1OTPffcM9ddd90a/fz7778/J598cnbffff84z/+Y7bZZptG2e8ZZ5yRiRMnplOnTo2yv4bYcMMNc8MNN6x03VNPPZXJkyfnc5/73Grte+edd86ECRPqte1hhx2WiRMnZrfddlutzwLWbiIVKNacOXNy2GGH5e///u/zxBNP5JhjjkmfPn2y7777ZuzYsfnud7+bs88+O2+++eYam2HevHlp165do+93s802y3bbbZcWLVo0+r4/zsEHH5xf//rXK13361//OgcffHA23HDD1dp3VVX13rZdu3bZbrvt0rp169X6LGDtJlKBYo0cOTKvv/56rrnmmmywwQYrrL/00kszceLEbLLJJrXLXnzxxXzjG99Iu3btsuGGG2afffbJE088Ued9759mnjNnTvr165dNNtkkW2yxRU499dS89957dbZJkn79+qWmpiYnnHBC7SnzD5+ifv/0/QddccUV6dGjR1q1apVddtklv/3tbz9y+4ULF2bQoEHp2rVrWrZsmW233TYjRozI8uXL62xXU1OTa6+9NldddVV69eqVNm3aZI899sjzzz9fr+/r0UcfnRdffHGF78t7772XW265Jd/61rdW+r5x48blK1/5Stq0aZNu3bplyJAhWbp0aZJk6tSpK/1+vf+1duvWLQsWLMjRRx+djTfeOIsWLar9Hk+dOjVVVaVv377Zbrvtsnjx4trPvPLKK7PBBhtk4sSJ9fragLWHSAWKdeedd6ZPnz7ZdtttV7p+o402ypZbbln7+o9//GO+9KUvZd68efmP//iP3HPPPdlwww3zla98JQ8++OAK7993333ToUOH/O53v8sFF1yQq6++OpdddlmS/zsVnSQXX3xxJk6cmGHDhtV79gceeCD9+/fPBRdckMcffzyHHnpozjjjjLz11lsr3X758uX5xje+kWuuuSaDBw/OY489lhNPPDGDBw/OySefvML2V155ZcaMGZOrrroqd955Z+bMmZOjjz56haBdme233z4777zzCkdTx44dm2bNmmX//fdf4T1PP/109tprr3Tu3DkPP/xwLrjgglxyySUZOXJkkqRTp04f+/0655xzssMOO+Sxxx5Ly5Yt66yrqanJddddlxkzZuRnP/tZkuTVV1/N+eefn4suuii9e/f+2K8LWMtUAIVq3bp1ddRRR9V7+3333bfq0qVL9c4779QuW7ZsWbXzzjtXvXv3rl02atSoKkl16qmn1nn/l7/85erLX/5ynWVJqlGjRtW+fuihh6ok1UMPPVRnuyFDhlQf/JH685//vNp4442rJUuW1C5bsGDBKre/+eabqyTVHXfcUWe/l112WZWkeuKJJ+rM1KFDh+qtt96qXXbttddWSapJkyat9HvzwdmnTJlS/eIXv6g22mij6t13361dv/fee1cnn3xyVVVV1bVr12rIkCG16/70pz9V/fv3r5YtW1a77NRTT6123333Op/x4e/X+19ru3btqtNPP73O8vf/d5gyZUqdr3eDDTaoJk2aVB1xxBHVl770pWrp0qWr/JqAtZcjqUCxli9fnvXWW69e27733nt54IEH8q1vfStt2rSpXd6sWbN873vfy8SJEzN58uQ67zn99NPrvO7evXtee+21Tz54kgMPPDCLFy/OAQcckMceeyxVVaVVq1ar3P7OO+9M+/bt841vfKPO8pNPPjk1NTW5++676yz/zne+U+da2e7duydJvef/1re+lUWLFtU+UmvatGl58MEH069fv5Vuv+222+ayyy5Ls2b/95+N7bffPq+++mq9Pm/evHkrPSL8YWeeeWZ22WWX7L///rn77rszevToev8dANYuIhUoVteuXfPKK6/Ua9s33ngjS5cuTdeuXVdYt9VWWyVJZs+eXWf5B2M2+VvQvn+N5SfVu3fvPP3002nXrl322muv/N3f/V0eeuihVW4/Z86clc7eunXrbLbZZvWaPUm95//c5z6XQw45pPaU/w033JBtt902X/rSl1a6/dtvv53Bgwdnp512ysYbb5z1118/Z511VoNulPq7v/u7j92mWbNm+cEPfpBXXnklBx544Cov9QDWfiIVKNb++++fP/zhD3nppZdWur6qqkyaNClJsummm2a99dbLjBkzVthu+vTpSf52R/0n9f5RvQ9f+zl//vwVtt1+++1z6623Ztq0adl+++1zwAEHrHS+JNl8881Xum7BggV5/fXXG2X2D/vud7+bRx99NNOmTcttt922yqOoSXLkkUfmsssuyyGHHJKxY8fmqaeeyo9+9KMGfV59joguWrQogwcPzhe/+MWMHTs2jz/+eIM+A1h7iFSgWAMHDkzbtm1zwgknZMGCBSusHz58eHr37p3JkyenVatW6du3b26++eY62y5fvjzXXXddtt5660Z5xun7N2q9H75J8te//nWF544uW7Ys8+bNq33Pr3/96yxcuDBPPfXUSve7//7757XXXsu9995bZ/m1116bqqpywAEHfOLZP+zrX/96OnXqlBEjRuSPf/xjjj322FVu+8gjj+TEE0/M0KFDs9dee2XHHXfMrFmzsmzZsjrb1dTU1OvmrVW58MIL8/rrr+e+++7LYYcdln79+mXhwoWrvT/gs0ukAsXq0qVLbrvttvzxj3/MrrvumhtvvDHPPfdc7bWT5513Xi699NL06NEjyd8eSTV37twceOCBeeCBB/LII4/kiCOOyLPPPptf/vKXjTJTz5498/nPfz7Dhg3LQw89lPvvvz/77bdfunTpUme7888/P7vttltuvfXWPP/88xk5cmRatGiRnXbaaaX7/eY3v5m+ffvm2GOPzXXXXZff//73GT58eM4777wcd9xx+fKXv9wo839Qs2bNcvzxx+fKK6/MvvvuW+dJCR+2xx575K677sp//dd/5YknnsjZZ5+dRx99NG+88Uad7Tp27Jh7770348aNW+kTFT7K448/nuHDh2fEiBFp165drrjiisyePTuDBw9era8P+GwTqUDRvva1r2XChAnp27dvLrjgguy666458sgj88Ybb+SRRx7JmWeeWbvtjjvumHHjxqV169Y59NBDs//+++ett97KQw89lK9//euNMk9NTU3+7d/+LZtttln222+/nHbaaTnllFNWuClo8ODBOeSQQzJw4MDstttuue2223LbbbfVBvWHNW/ePPfcc0/69euXoUOHZvfdd88111yToUOHZtSoUY0y+8p897vfTVVVtc80XZXrr78+O+20U44++ugcfPDBWbJkSR588MEsWrQoL7zwQu12w4YNy4MPPpijjjoqL7/8cr3nWLBgQY4//vjsvffe+eY3v5nkb0eghw0blssuu8xpf1gH1VQNueodAAA+BY6kAgBQHJEKAEBxRCoAAMURqQAAFEekAgBQHJEKAEBxRCoAAMVp3tQDNKZWO53R1CMANKrXxl/e1CMANKqNW61Xr+0cSQUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA4ohUAACKI1IBACiOSAUAoDgiFQCA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T5JcfPHF+c53vtPowwEAsG6qqaqqauib3nnnnay//vpZf/31kySzZ8/OpptuWvu6qbTa6Ywm/XyAxvba+MubegSARrVxq/XqtV2Dj6Qmfzvl/0EdOnRYnd0AAMBK1StS99xzz9TU1Hzsdg8++OAnHggAAOoVqV/96lfX8BgAAPB/6hWpQ4YMWdNzAABArdX6tajvvvtuxowZk5/85CdJkocffjircf8VAACsVIMj9YUXXsi2226bkSNH5sorr0yS/OY3v8nFF1/c6MMBALBuanCknnXWWRkyZEieeeaZtGrVKkkyfPjwjBkzptGHAwBg3dTgSH3ppZfyve99L0lq7/hv27ZtFixY0LiTAQCwzmpwpG600UZ5/vnn6yybMGFCNtlkk0YbCgCAddtqne7/+te/nssuuywLFy7MTTfdlMMPPzwDBw5cE/MBALAOavBvnDrllFOyZMmS/OIXv8ibb76ZYcOG5bzzzssJJ5ywBsYDAGBdVFOtRc+OarXTGU09AkCjem385U09AkCj2rjVevXarsFHUpPktddey29+85tMnTo13bt3z9FHH51OnTqtzq4AAGAFDb4mddy4cendu3duvPHGzJ07N7fcckt22GGHPP7442tiPgAA1kENPpI6YMCADBs2LKeeemrtsuuvvz79+/fPk08+2ajDAQCwbmrwNakdOnTI7NmzV1jesWPHvPrqq4022OpwTSqwtnFNKrC2qe81qQ0+3b/ddttl4sSJdZbNmDEjXbt2beiuAABgpep1uv+RRx6p/edjjz02hx12WH784x+nffv2eeutt/KjH/0oP/3pT9fYkAAArFvqdbq/WbOPP+BaU1OTZcuWNcpQq8vpfmBt43Q/sLZp1EdQLV++/BMNAwAADdHga1IBAGBNa/AjqJYtW5Zrrrkmzz77bJYsWZIkWbBgQV544YW88MILjT4gAADrngYfST3jjDNy2WWXpU2bNrn99tuzxRZb5Kmnnsq//Mu/rIn5AABYBzX4OaldunTJ888/n0022STdunXL1KlTM2HChAwePDh33HHHmpqzXtw4Baxt3DgFrG3W2HNSk6RNmzZJkpYtW+bdd9/N5z//+TzzzDOrsysAAFhBgyN13333zTHHHJOlS5emT58+GT58eO688860aNFiTcwHAMA6qME3To0cOTIjRoxI8+bNc8455+RrX/taFi9e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FKemqqqqqYeAz5JFixZl2LBhGTRoUFq2bNnU4wB8Yn6uUSKRCg00f/78tGvXLvPmzUvbtm2behyAT8zPNUrkdD8AAMURqQAAFEekAgBQHJEKDdSyZcsMGTLEzQXAWsPPNUrkxikAAIrjSCoAAMURqQAAFEekAgBQHJEKDTR69OjssMMO6dy5c3bbbbeMGzeuqUcCWC3Lly/PE088kbPPPjubbrppRo8e3dQjQS2RCg1w44035p//+Z9z2223ZebMmfnhD3+YAw88MFOmTGnq0QAabNSoUTnzzDPTqlWrrLfeek09DtTh7n5ogF69euXUU0/NwIEDa5d94xvfSK9evXLppZc24WQAn0y3bt0ydOjQnHDCCU09CiRxJBXqbcaMGZk0aVIOOuigOssPPvjg3HvvvU00FQCsnUQq1NOsWbOSJB07dqyzvGPHjrXrAIDGIVKhnlq0aJEkadas7r82NTU1cdUMADQukQr11Llz5yTJq6++Wmf5q6++mk6dOjXFSACw1hKpUE/t27dPnz59cs8999RZft9992W//fZroqkAYO0kUqEBfvjDH+aSSy7Jyy+/nCS5/fbb87vf/S5nnHFGE08GAGuX5k09AHyWfPOb38z8+fNz0EEH5Z133kmnTp1y1113pWfPnk09GgCsVTwnFQCA4jjdDwBAcUQqAADFEakAABRHpAIAUByRCgBAcUQqAADFEakAABRHpAIAUByRCqw1TjjhhLRp0yadO3dOp06dss0222TQoEF59913G/Vzpk6dmpqamkydOjVJcuutt6Zz586N+hkfZ+jQofnqV7+6yvXdunXL6NGjV2vfNTU1efjhh1frvcnHzwZQHyIVWKsceeSRmTlzZmbNmpX77rsv999/f/7pn/7pU/nM+jrppJPy6KOPrsGJAD77RCqw1urevXvOO++8/Pa3v23qUeq4//77s2zZsqYeA6BoIhVYq7377rvZYIMNkvzfafpJkyZljz32yDnnnFO7/PDDD0/nzp3To0ePXHTRRXUicty4cdltt93SoUOH7LrrriucCh89enS6detW+3rZsmUZNmxYevXqlU6dOuX//b//lyeeeCJJsu+++2bmzJk58sgj07lz57z00ktJkrFjx6ZPnz7Zcssts+uuu+aRRx6p3d/y5cvzk5/8JF27dk2XLl3y7W9/O2+88cZqf0/efvvtnHTSSbWXRZxyyilZsmRJnW0mTZqUvn37pn379tlll13y2GOP1Vl/1VVXZbvttkvHjh2z5557ZsKECav8vDFjxmTrrbdO+/bts88++2TixImrPTuwDqkA1hLHH398dfzxx1dVVVXLli2rxo8fX/Xo0aM655xzqqqqqilTplRJqpNOOqmaOXNmVVVV9c4771RdunSpTj311Grx4sXV7Nmzqz59+lTDhw+vqqqqZsyYUbVt27a66qqrqqqqqr/85S/V3nvvXSWppkyZUlVVVY0aNarq2rVr7Rw/+MEPql69elWTJ0+uqqqqxo4dW2299dbVokWLqqqqqq5du1YPPfRQ7fZ33HFH1apVq+p3v/tdVVVVddddd1Vt27atpk2bVlVVVV1++eVV586dq0mTJlVVVVX3339/tfHGG1d9+/Zd5feia9eu1ahRo1a6bujQodUhhxxSLViwoHr99derrbfeurryyitr1yepevXqVU2YMKGqqqr65S9/WbVt27b2e/aLX/yi2nzzzatnnnmmqqqquuqqq6qOHTtW8+fPr6qqqoYMGVI72/z586vmzZtXf/7zn6vly5dXY8aMqX0fwEcRqcBa4/jjj6/atGlTde3aterSpUu12267VSNGjKgWL15cVdX/RepNN91U+54bbrih2nTTTWsDsqqq6vbbb6+Nzp/97GfVLrvsUudznnvuuVVG6qJFi6rWrVtXt912W533fHD/H47UvffeuzrjjDPqbH/ooYdWQ4YMqaqqqnr37l0nIquqqgYMGLDakVpVVe33pKqqqn///tVJJ51U+zpJdfPNN9fZvk+fPtUll1xSVVVV9ezZszbi37fjjjvWft4HI3XhwoVVhw4dqgsvvLB66623VjkPwIc1b8qjuACN7YgjjvjYu9q/+MUv1v7zzJkz8+6772abbbapXbZ8+fK88847WbRoUSZPnpwePXrUeX/btm1Xue+5c+dmwYIF6d27d53l66+//irfM3PmzDz77LO58847a5ctXLgwn/vc55KkwTN8nKeffjqXXHJJnnvuuSxatCjz5s3LoYceWmebDh061Hm97bbb1j7NYObMmfn5z3+eK664onb9ggULMn369BU+q2XLlhk/fnwuuuii9OzZMwcffHBGjhyZdu3arfb8wLrBNanAOme99dar/eeePXumU6dOmTp1au2f6dOn54033kjLli2z5ZZb5uWXX67z/tmzZ69y35tvvnlatWpVe63p+z58zecH9ezZM2eddVadGWbPnp1rr702SRo8w0dZvHhx9t577/Tu3Tu///3vM23atHz/+99fYbu33nqrzuvJkyfXXnfbs2fP/PznP68z75w5c3LhhReu9DO7d++eUaNG5c9//nP+8pe/5Lzzzlut2YF1i0gF1mkHHXRQ1ltvvZx//vlZtGhRkuTBBx/M0UcfnSQ59thj8/LLL+dXv/pVqqrKlClTMmDAgFXub/3118/pp5+e8847L5MmTUqS/P73v88222yTuXPnJklat26dOXPm5M0330yS9O/fP5dffnntzVILFy7MoEGDcsMNNyRJTjnllAwfPjwvvfRSqqrKzTffnFtvvXW1vt6lS5dm0aJF+cIXvpC2bdtmwoQJ+fd///csWLCgznY/+tGPao+MXn/99Xn55ZdzzDHHJEkGDBiQoUOH5vnnn0+SzJ8/P9/73vfy4IMPrvB5r732WgYMGJC5c+emXbt22WmnnTJv3rzVmh1YtzjdD6zTWrVqlfvvvz/nnntuevbsmZqamvTu3TvDhw9PkvTo0SMPPPBA+vfvn6FDh2arrbbKiBEj0rdv31Xu86c//Wk222yz7L///nnvvffSoUOHXH311dl8882TJKeddlpOO+20bLXVVrnrrruyzz77ZNSoUTn77LMza9asbLDBBjn00ENzxBFHJEnOPffcJMk+++yTJUuWZL/99sugQYNy1113feTXNnDgwAwePLjOsgkTJuT6669P//79c9ppp2XnnXfOiBEjcv7556/wNRx11FGZNm1aOnfunLvvvjtdunRJ8rfnvDZr1qz2KQOtW7fOcccdt9LvyaabbpoWLVqkT58+adasWXr06JExY8Z85NwASVJTVVXV1EMAAMAHOd0PAEBxRCoAAMURqQAAFEekAgBQHJEKAEBxRCoAAMURqQAAFEekAgBQHJEKAEBxRCoAAMURqQAAFOf/B5TVuBXJIdhKAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "# 混同行列の計算\n",
    "matrix = confusion_matrix(y, y_pred)\n",
    "\n",
    "# ヒートマップの表示\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)\n",
    "plt.xlabel('Predicted Labels')\n",
    "plt.ylabel('True Labels')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 特徴量の重要度(フィルム切れの要因)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA88AAAIgCAYAAAC77lsgAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAByKElEQVR4nO3deXwNZ///8ffJIhKRqJ2ESO1LNWopSm0pqim6aKktJa0qyh0Urb16c6Noa6tQsVTtS1vU0lLEElF77WtQrTURsmd+f/jlfJ1mmSSNRvT1fDzO42tmrrnmM8fc9X2fa+Yai2EYhgAAAAAAQJrscroAAAAAAAAedYRnAAAAAABMEJ4BAAAAADBBeAYAAAAAwAThGQAAAAAAE4RnAAAAAABMEJ4BAAAAADBBeAYAAAAAwAThGQAAAAAAE4RnAAAAmNq6dassFou2bt2a06UAQI4gPAMA8IgIDg6WxWJJ83P9+vVsP+Yvv/yi06dPZ3u/WZGbw9nq1at148aNnC4jS0aOHGlzneXNm1fly5dXnz59dPXq1Ww5RkxMjBYuXJgtfQFATnHI6QIAAICtzZs3y8PDI8X6J554IluPc/HiRTVu3FhbtmxRuXLlsrXvf5Nt27bplVde0blz51SoUKGcLifLjh07Jkm6efOmwsLCNHbsWK1bt06HDh1Svnz5/lbf48eP19dff61OnTplR6kAkCMIzwAAPGLKli2rMmXKPPTjJCUlPfRj/Bs8Lt9jpUqVrH+uX7++nnrqKTVt2lSrVq3626H3cfmOAPy7cds2AAC5TEJCgkaOHCkvLy+5uLioRo0aqd4S+/3336tWrVpycXFRhQoVNHXqVOu2rVu3ytvbW5LUpEkTWSwWjRw5UpLUuHFjNW7cOEV/D7aR7t/uW6ZMGd27d09vvvmmChQooNjYWEnSmTNn9Oqrr8rNzU1FihRR27Ztdfz48Uyfq7+/vxo0aKDly5fL29tbrq6u6tq1q+Li4rRo0SJ5e3vLxcVFb775pu7cuWPd7/z587JYLNq/f7+GDBmi4sWLK1++fGrbtq3Onz+f4jjBwcF66qmn5OTkpJIlS6p37966ffu2TZvGjRvL399fp06dUt26deXr66vg4GA1adJEkuTt7S2LxaLg4GBJkmEYGj9+vJ588knly5dPtWvX1pYtW1I9v8OHD6tZs2bKly+fvL299eWXX6ao8eeff1aDBg3k7OysIkWKqHPnzvrzzz9t2mzevFn16tWTs7OzvLy81KdPH926dSvT37sk1a5dW9L9OxTSY/bd+fv7a9SoUbpw4YL11vDU/g4A4FFHeAYAIJfp0qWL5s+fr3Hjxmn79u3q3LmzAgICNHHiRGub1atXq02bNmrYsKG2b9+ud955Rx988IGWL18uSapTp442b94sSZo3b56OHTum3r17Z6megQMHqlq1atqxY4ecnJx06dIl1a9fX/b29vrhhx+0atUq2dnZqW7dujpz5kym+z937pyGDRumWbNmacaMGVq0aJE6deqkUaNGadasWZo3b57Wrl2roUOHptj3vffe04ULF7RmzRotWLBABw8e1PPPP2/zfPInn3yigIAAtWnTRtu3b9fEiRP13XffqVGjRrp3755Nf0lJSXr77bfVt29fff3113rllVc0b948SfeD67Fjx/TKK69IkgIDAzVixAgNGjRIW7duVaVKlfTSSy8pPDzcps9Lly7p5ZdfVvv27fXLL7+oWbNm+uCDD7Rz505rm/Xr16t58+YqUaKENm3apEWLFunYsWNq3Lix9QeL9evXy8/PT76+vvrll180depU/fLLL2ratKkSEhIy/b3/+uuvkiRPT88022Tkuxs7dqx69eqlkiVL6tixYzp27FiqjyUAwCPPAAAAj4S5c+cakgx7e/sUn6+++sowDMPYvn27YbFYjBMnTtjs+7///c9wc3Mz4uLiDMMwjNDQUGPo0KE2bV588UXjrbfesi6fO3fOkGRs2bLFpl2jRo2MRo0apahPkjFixAjr8ogRIwx3d3ejV69eNu26d+9uNGjQwEhKSrKuS0xMNKpWrWr07NkzzfPfsmVLinq6du1qSDL2799vXde0aVNDknHw4EGbdp6eninOrUWLFjZ1/Pbbb4adnZ0xbNgwwzAM4+LFi4aDg4Pxn//8x6aWffv2GXZ2dsa4ceNsvpeiRYsaCxcuTLXuc+fO2axftGiRMXfuXOtyTEyM8cQTTxizZs1KcX5Lliyxrrt3756RJ08em7+/cuXKGbVq1TISExOt6/744w9j8ODBRmRkpLXNxx9/bFPD5cuXDXt7e5v+/2rEiBHGg/8v4fXr140VK1YYXl5eRvHixY3bt2/bnGfy309mvrsRI0YYXl5eadYAALkBzzwDAPCIWbdunUqWLGmz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e/fueuqppzK0z82bNxUQEKAjR45o8+bNKWan/uKLL7R69WpVrVpVVatWVUBAgEqUKCEHBwdNnz49U/UlJSWpY8eO8vb21vvvv59i+44dOzRgwADrcvPmzTVp0iTNmjVLH330kWbMmKGlS5eqXr168vX1TbH/rl275OTkZF0+evRoilHvy5cv24TbiIgI1alTRz/88IMKFy6srVu3qly5cjp9+rT1lvOFCxfqxo0b1n1u374tNzc3zZo1S2PHjtXvv/9uM8v27du3de/ePV28eFGurq7pfidOTk42NQMAAABAZhCeM+nWrVt67bXX5ODgoIkTJ5q2T0pK0qJFi9S/f381aNBAe/futXnWOdnIkSNtngGW7o+8jh8/PlP1xcbGqnPnzjp06JD27NmT6ih1RESEGjdurKlTp2rHjh3W28cbNGigl156Se3atdPZs2f1888/p3qM5cuX25xD1apVTd9JPXPmTNWtWzdTt7ifOXNGJUqU0Lvvvqtu3brJz89Pbdq0Uc+ePXX79m01btxYAwcONA3OAAAAAPB3pUxWSJVhGFqyZImqV68uOzs7rV+/Pt3JxCRp9erVqlChgoYOHaqvvvpKK1asSDU4p6VmzZo6dOiQ4uPjM9R+9+7dqlWrlg4dOqTt27enuN3ajGEYyps3r3bu3KnExETt378/RZurV69q9erVunDhgqZNm5ahfiMiIjRp0iT17ds3U/Xs3LnTOhmbg4ODVq5cqV27dun1119XgwYN9J///EcdO3bMVJ8AAAAAkBWE5wwIDQ1V+fLl1bNnT/Xv318bNmzQE088ke4+9evX19GjRxUYGKhjx46pbdu2mT6ut7e3XFxcdOTIEdO2q1atUoMGDfTcc88pLCxMXl5e6bb/9ttvVa1aNXXp0kWGYWj9+vVq1KiR5s+frwMHDmjcuHF64403VL16dX3//ffW/fr166eOHTtq7dq1mjFjht5//31FR0fr/PnzioiIUHR0tO7evasbN27o3r17kqThw4eratWqql+/fobPPTo6Wt9++631e4uPj9fOnTt19+5dXbhwQU5OTlqwYIG++OIL7dq1S3fu3Mlw3wAAAACQWdy2nQE1a9bUmDFj9PLLLytfvnwZ2mfSpEn67LPPbN4LnRUTJkxQwYIFTdu98sorOnXqlLy9vTPUb4cOHayvq0pKStKzzz6r5s2ba+jQoXJ2dpafn59OnDihMWPGWF9DNWfOHB08eFC7d++Wu7u7QkND9eWXX+q7775TnTp1dPPmTSUmJkq6P5nXL7/8oueee04HDx7U5MmTtWvXLrVr107S/VdQPfh/GzRoYK3N1dVVTZo0UYMGDVSuXDk1bNhQp0+fVpMmTdSjRw+98MILslgs2rFjh5YvX67Zs2erWrVqWrRoUQa/VQAAAADIHIuR2Xf+4LGUmJhoOnt1ZGSkbty4kW5AT0pKkiSbZ61jY2OzNFlX8quqLly4oFKlSqX6/HZWRUZGyt3dXaX6LZWdU/q33wOPs/PjXsrpEgAAAHJUcjaIiIiQm5tbmu0YeYakjL32yc3NLd2LSVKqATers1wn92V2CzoAAAAAPGw88wwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGDCIacLAHLSkVEt5ObmltNlAAAAAHjEMfIMAAAAAIAJwjMAAAAAACYIzwAAAAAAmCA8AwAAAABggvAMAAAAAIAJwjMAAAAAACYIzwAAAAAAmCA8AwAAAABggvAMAAAAAIAJwjMAAAAAACYIzwAAAAAAmCA8AwAAAABggvAMAAAAAIAJh5wuAMhJ1UZskJ2TS06XAeARcH7cSzldAgAAeIQx8gwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPuUxcXJyOHj2a02UAAAAAwL8K4fn/+/7773Xy5MmcLiNdx48fV1hYmDp27KidO3fmdDlW/v7+mjlzZrptDMNQfHx8thzv9ddfV3BwcLb0BQAAAAAZ8a8Kz5cuXVJMTEyK9Z999plcXFzUunVrXbp06aHWYBhGlvZJSkrS1atXFRAQoC+//NI0rIaFhSk2NjbN7VOnTlWlSpWsn+vXr0uSatWqpRIlSqhkyZIqU6ZMpmtNS3BwsJo2baqbN29Kkuzt7VWtWjWbT9GiRTV16tRsOyYAAAAAZJdHKjwbhqHExETFx8crNjZW0dHRqYbdrBoyZIgqVqyo9evX26x3c3PT6NGj9emnn+rixYs223r16qXg4OAshd7UJIfItBQqVEh58uSRg4OD7OzsZLFYZGdnJ0dHR928eVP9+/fXpUuXNH/+fJv9rl+/Ln9/f+ty7dq1FR4enuZxevfurePHj1s/hQsXliQlJSVp3759OnToUKr7DRgwQGXKlLH5LF++XB9//HGK9d988411vy5duqhixYpq3bq1JMnZ2VlHjhyx+bz77rspjjdmzBgVLFhQ5cqVs34OHDigMWPG2KwrV66ctmzZkub5AgAAAMDf4ZBTBw4ODlaPHj2UlJRk/aQmT548Onr0qMqVK5ctx/z888/Vtm1bbd26VfXq1ZMkvfPOO7p27ZoaNmyookWLWtsbhqHy5curd+/eWrJkiRYuXKhChQrZ9BkZGSnpfgDPiF9++UUlS5ZMc/vRo0dlsVhkb29v/cTExKhy5coqXLiwXn311VT3i4iI0IIFCzJ0O/Ply5fVqFEjm3VVqlTRd999p2vXrqlgwYKKioqybtu2bZsaNmwoi8Wi69eva/DgwXrvvffSPYa/v7/u3LljXba3t9esWbMUGhoqSYqOjla1atVs9vnzzz81fPjwFH198MEHGjlypEJDQzV//nzr6PTy5ct19uxZffjhh6bnDAAAAAB/R46F5zfffFOtWrWyBkQ7OzubP8fGxsrX11c1a9b828H58uXL+vnnn9WpUycFBgbqpZdeUsWKFW3a9O3bV/v27bMJzxaLRf369VOrVq3UunVrzZkzJ0VQmzBhgsLDwzMUWuPi4vTdd99pzpw5abYpXrx4inVBQUEqXbq0nn/++TT3i4+PV548eUxrkCQPDw9t27ZNV65csa7z8fFRSEiIihUrprx58yoqKkrR0dGKiIjQK6+8oj/++EMODg4qWLCgXF1dTY/xYLvExES1b99egYGB1h8skkeeHzR06NBU+5o6daoWL16se/fu6datW9q8ebOk+z8YxMXF6euvv5Z0/+/i5ZdfTrWP2NhYm9vYk3/0AAAAAICMyLHw7OzsLGdn51S3JSUlqX379ipcuLC+/PLLv32skydPatCgQZo+fbq++eabFMFZuh+eL1++nGpArVChgvbu3at8+fLZrD979qwmT56stWvXZqiOJUuWyMnJSX5+fhmuPTIyUhMnTtS8efPSbXf37t0MhdpkoaGhWr16tXX5008/VZcuXTRr1ixJ98PvU089pRo1aqhNmzZycLh/qUyaNEnS/dvCf//9d0n3Q6y9vb31+PXq1dOyZcusfdvb2+udd95R69atNX/+fL344ouZGnnu06ePRowYoa1bt2rKlCnWumfPnq3jx49r4sSJpuc7duxYjRo1KoPfDgAAAADYyrHwnOzGjRv6+OOP9cknn6hIkSJKSEhQt27ddO3aNa1fv1729vbWtmXKlNGFCxcy1O+ECRM0YMAASVKTJk104sQJde/eXUFBQRo7dqxN25UrV2rZsmVpPucrSfnz57dZjo2NVadOnfTaa6+luAU6NbGxsfrkk080aNAgOTo6ZugcJKlbt25q1qyZmjdvnm67GzduWJ9bTk1cXJwCAgI0a9Ys5c2bV4ULF1alSpWs2wsUKKDevXvr2rVrWrx4sSQpICBALi4uqY7m7t271/rnfv36qUyZMurXr1+ax2/evLnWrFmjUqVKSbo/Gp0RSUlJsrP7v0fzd+/ebf3xITw8XC+88EKG+hkyZIgCAwOty5GRkdZaAAAAAMBMjodnNzc3RUREqF69evr+++81ePBgJSQkaOPGjSlGpnfv3q2EhIQM9VugQAGb5fz582vp0qUpQtvq1avVqVMnzZ49W15eXhnqOy4uTh07dlRkZGSGR8aHDh0qJycnvf/++xlqL92fBXz16tV6/vnnde3aNRUpUiTNtidOnFDZsmXT3B4YGKhDhw5ZJz4rXry4fHx8rNsdHR0VFRWlnj176qOPPpLFYtGZM2e0YcMGtWzZMsUt4TNmzFBCQoL69OmT4fOpVauW7t27pylTpmj27Nlpttu7d6/17z4+Pt464l+3bl3t3r3bpq27u3uGju3k5CQnJ6cM1woAAAAAD8rx8Ozo6KhvvvlGffr0UbVq1dS6dWutWbPGepvwg1J7HjizHhzJnjlzpvr166eZM2fqrbfeytD+ly9fVufOnXXlyhVt2rQpQxOFzZo1S1999ZVCQkIy/FzyqFGjNHXqVP3666+aNGmSatasqeXLl6tOnTqptl+zZo2aNWuW6rbhw4dr1apV2rFjhzWUTp48Wb/88ou1zfTp0zVs2DCFh4crf/786tGjh1q2bKn//e9/qdb8008/qU2bNjbHGDdunHX59ddfT/HaqYULF+rXX3+Vq6urBgwYoMKFC2vz5s0aMGCAOnXqpK1bt6pcuXI2P3Dcvn1bJUqUUKtWrbRr1y7lzZvXZtK2c+fO6a233lJQUFB6XycAAAAA/C2PxKuq7OzsNG3aNHXv3l0HDx7UjRs3HurxwsLC1KBBAw0ZMkSLFy+2ecVTWiIiIjR27FhVq1ZNrq6uCgkJMb3tNz4+Xh999JH69++vlStX6qmnnjI9zrVr19ShQwfNmjVLmzdvVvXq1RUcHKzu3burSZMmKV6zJUmbNm1SSEiIunbtarP+zp076tq1q4KDg7VlyxZ5e3tbt124cEHBwcE6cuSI6tata51Aa/LkyZo/f77at2+vSpUqqV27dimOFxUVpR9//NFm1Hf06NG6evWq9fPX4GwYhj777DO99NJLpt/Bg86cOaMSJUpo3bp12rZtmwoVKqRVq1bpyJEjmjRpkooVK6YxY8Zkqk8AAAAAyKxHIjwnmzZtmry8vOTn52czM3J2uHz5sqZNm6ZmzZqpbt26qlixok6ePKm2bdumu9+sWbP0yiuvqHjx4lq4cKGCgoL03XffpXhl1V+tXLlSNWrU0PLly7V161b5+vqm2z4uLk7//e9/Va5cOf3xxx8KDQ3V008/bd0+YsQIDR48WG3atNH3339vXX/w4EF16NBBn376aYpXYPn6+ur8+fMKCwtThQoV0j1+snv37snNzU1bt25VZGSkzYzcyb7++muVLVtWffv21ZkzZzLU75IlS5SQkKCWLVtmqL10/3nn3bt3W7+Hp556SgsXLtQbb7yhnj17qn///vrpp59UrFixDPcJAAAAAFmR47dtP8jR0VErVqywzkqdnf73v/9p7dq1at++vaZOnarKlStnaD+LxSIvLy9t2LAh3VdF/dXly5f15ptvqn///nJxcTFt7+DgoMuXL+vzzz9X165dZbFYUrQZNmyY3N3dVbt2beu6ESNGqEePHjaTYUlSr169VL58efXp08dmwq0HtW/fXnnz5tWVK1fUpEkTTZgwQZ999pm6d++uNWvWKDAwUN7e3nrxxRc1a9YsFS1aVFeuXNGoUaO0fv16hYeH69lnn1XJkiXVqlUrXb16Va6urrK3t1d8fLxiYmJUpEgR3b17VwMGDNB///vfVM8rLatWrZKnp6eefPJJSfcnRQsJCVFCQoIuXbqkmJgYffLJJ2rRooWefvpplS9f3ua2fAAAAADILhYjeQapx1xCQkKqz1HndoZhZCqQJvPz89PIkSNVq1YtSfdfXdWxY0dNmTLF5tbqkJAQBQUFae7cubJYLGrYsKEaNmyo//73v5Kk3377TZMnT1ZISIguXbqkqKgo66RklSpV0rFjx3T06FH16NFD27dvV//+/bV06VLT+k6ePKmvvvpKRYoUkZubm/r37y9HR0e98sorCggIkLe3t+7du6dly5Zp/fr12rJli+bNm5fhke3IyEi5u7urVL+lsnMy/3EDwOPv/LjMPVYCAAAeD8nZICIiIt05rf414RnmEhMTTUduDx06pKeeeirdwJ5aP7GxsVm6myApKUmxsbHWicOyC+EZwF8RngEA+HfKaHh+/IZikWUZueW5evXqWeonq7fh29nZydnZOcVrywAAAADgn/RITRgGAAAAAMCjiPAMAAAAAIAJwjMAAAAAACYIzwAAAAAAmCA8AwAAAABggvAMAAAAAIAJwjMAAAAAACYIzwAAAAAAmCA8AwAAAABggvAMAAAAAIAJwjMAAAAAACYIzwAAAAAAmCA8AwAAAABggvAMAAAAAIAJwjMAAAAAACYIzwAAAAAAmCA8AwAAAABggvAMAAAAAIAJwjMAAAAAACYccroAICcdGdVCbm5uOV0GAAAAgEccI88AAAAAAJggPAMAAAAAYILwDAAAAACACcIzAAAAAAAmCM8AAAAAAJggPAMAAAAAYILwDAAAAACACcIzAAAAAAAmCM8AAAAAAJggPAMAAAAAYILwDAAAAACACcIzAAAAAAAmCM8AAAAAAJggPAMAAAAAYMIhpwsAclK1ERtk5+SS02UAAAAA/xrnx72U0yVkCSPPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8AwAAAAAgAnCMwAAAAAAJgjPAAAAAACYIDwDAAAAAGCC8JwJhmHIMIycLuORM3ToUA0YMMC03bhx4zRx4kTr8pYtW9StW7eHWRoAAAAAZIt/VXgOCQnRpUuXsrx/hQoVdODAgewrKIMaNWqklStXZmnfw4cP65NPPslw+7CwMMXGxmbpWJk1f/58hYSEyMfHR66urtq8eXOq7YKDg1W8eHHVrVs3xad27doqU6bMP1IvAAAAgH+vXBueExMTFRsbq3v37unOnTu6ffu2rl+/rj/++EPXrl1LdZ/+/funGdAy4sknn1RYWJjNutRGohMSEjR06FCdPHky1X4iIiL03HPP6ffff8/QcR0dHRUfH5/peq9evSo/P790w/D169fl7+9vXa5du7bCw8MzfazMCg8P186dO3XkyBGtW7dOHh4eev7559NsX758ebVv3z7F59VXX33otQIAAACAQ04XkBW+vr766aefrMt2dnZydHSUg4ODHB0dFRUVpQ0bNqhp06bZdsy7d+/K09NT3377rU6dOqXDhw/ryJEjeueddzR8+HCbtg4ODgoNDdX+/fu1du3aFH2FhobqwIEDKly4cIaO7ejoqLi4uEzVe/XqVbVs2VINGzZMd+Q5IiJCCxYsUHBwcIb6bdOmjfbs2ZPqtoULF9oslypVSnv37tWuXbv0wgsvWH8AmDt3ripVqqTr16+rdu3aunXrlu7evas6depIkmbPnq1atWrZ9HXmzBktX748xTETExMzVDcAAAAA/B25Mjx/9913MgxDjo6OcnR0lMVisW5LSEiQu7u73Nzc/vZxfv/9d3Xp0kXHjx9XVFSUihUrpmvXrumll15SkyZNVLVqVZUuXTrVfcePH68aNWpo165dqlevns22tWvXqlGjRnJ0dLRZ3717d82bN89mnWEYSkpK0saNG/X2229LkurWrasdO3akWffJkyfVsmVLPfvss5o/f77N9/NX8fHxypMnT7rfw4PWrFlj/fOIESPk6+urhg0bSpJWrVql9evXa/LkycqXL5+1Xb169RQVFaVx48bJwcFB+fPnV+/evRUUFKSTJ0/queee04wZMzR16lQNHTpUUVFRkqQrV65o586dcnV11RdffJFuXcuXL5eLi4tatWqV4XMBAAAAgIzKleHZxcUlzW27d+9Wnjx59PTTT2eqz/j4eN27d0/u7u7WdYUK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xWKxrjt8+LAkKX/+/A+3WOQ6WflvWMmSJa1/PnHihCZMmKDGjRs/zDKRi2X138mePXvKz89P9evXf9glIhfL7PUVFxena9euqXTp0tlyfMIzHjuXL1+WZPuPffJy8ra/tv9rW0ny8PBItT3+3TJ7fQGZ9XevsX379qldu3by9/fnGXukkNXra8qUKXJzc5OPj4+eeeYZzZs376HWidwrK9fYggULtH//fk2YMOGh14fcLbPX15UrV5Q3b1599dVXqlGjhp588kl17Ngxy/OCEJ7x2EkeQbazs728LRaLDMNItf1f26bXHv9umb2+gMz6O9fYF198oYYNG8rf31+zZ89+aDUi98rq9dWvXz/dvn1bP//8sw4fPqxt27Y91DqRe2X2Gjt//rz69eunBQsWyMXF5R+pEblXZq+viIgIFSlSRCVKlNDOnTt1+PBhFS5cWE2bNtXdu3czfXzCMx47np6eku7/0vSgK1euyMPDI9X2f22bXnv8u2X2+gIyKyvXWFJSkgICAjR16lRt2bJFn376qezt7R96rch9/s5/w+zs7FSvXj19/PHH6tSpk+Lj4x9anci9MnONJSUlqXPnzurTp4/q1Knzj9WI3Cuz/w17+umndeHCBXXq1EnOzs7Kly+fJk2apKtXr2r79u2ZPj7hGY+dYsWK6emnn9a6dets1m/YsEEtW7ZM0b5FixbatGmTEhISrOuOHj2qa9euqWnTpg+9XuQumb2+gMzKyjU2aNAgnThxQmFhYXr22Wf/iTKRS2X2+jpz5ox+++03m3WFCxfWnTt3FBUV9VBrRe6UmWssMjJSO3bs0KhRo2SxWKwfSfL29laDBg3+sbqRO2Tl38ikpCSbZcMwlJSUZDNXSIZly7RjwCNm0aJFhoeHh3HixAnDMAxj1apVhpubm3H69OkUbePj442qVasagwcPNhISEozbt28bTZo0MXr06PFPl41cIjPX14OYbRsZlZlrbPfu3UbhwoWN69ev/9NlIpfKzPU1YsQIo3z58taZaiMiIgw/Pz/jueee+0drRu6S1X8nk4nZtpGOzFxfISEhRrly5YzQ0FDDMAwjOjraeP/9943y5csbMTExmT424RmPrZkzZxrly5c3SpQoYdSqVcvYtm2bYRiGER4ebnh4eBhLly61tg0PDzdat25tlChRwvDw8DD69euXpf9B4d8jM9dX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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 特徴の重要度を取得し、可視化\n",
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.barh(x.columns, model.feature_importances_)\n",
    "plt.xlabel('Feature Importance')\n",
    "plt.ylabel('Features')\n",
    "plt.title('Feature Importance Plot')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### フィルムの断裂予測\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.0166612], dtype=float32)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# データを含むDataFrameを作成\n",
    "a = pd.DataFrame([[30, 0,200 ,0,131,135]], columns=['マージンオイル補充量', 'マージンオイル交換量', 'フレキソオイル補充量', 'フレキソオイル交換量','マージンノズル設定温度','フレキソ設定温度'])\n",
    "\n",
    "# 予測を行い、クラスラベルを取得する\n",
    "y_score = model.predict_proba(a)[:, 1]\n",
    "y_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "今回の条件で、フィルムが起きる確率は98.3%です\n"
     ]
    }
   ],
   "source": [
    "y_score = model.predict_proba(a)[:, 1]\n",
    "percentage = y_score * 100\n",
    "percentage = round(percentage.item(), 1)  # 必要な桁数に調整\n",
    "desired_percentage = 100 - percentage\n",
    "print(f\"今回の条件で、フィルムが起きる確率は{desired_percentage:.1f}%です\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  },
  "vscode": {
   "interpreter": {
    "hash": "63cb323edc9dfd3801db5ba0fed665ec16605e8bb909d22883095c9fca3667ab"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
