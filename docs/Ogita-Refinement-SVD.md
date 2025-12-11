# 実装用レポート：SVD に対する Ogita らの反復改良法

本レポートは、Ogita & Aishima (2019) に基づく特異値分解 (SVD) の反復改良法 (Iterative Refinement) を `eigenpairflow` に実装するための数学的仕様書である。特に、論文中で明示的な式が省略されている「特異値が縮退・近接している場合」の数値的に安定な解法について、厳密な導出と実装指針を提供する。

## 1\. 問題設定と基本戦略

対象とする実一般行列 $A \in \mathbb{R}^{m \times n}$ (ここでは簡単のため $m=n$ とするが、一般の場合も同様) に対し、近似的な SVD が与えられているとする。

$$
A \approx \tilde{U} \tilde{\Sigma} \tilde{V}^T
$$

ここで、$\tilde{U}, \tilde{V}$ は近似的な直交行列、$\tilde{\Sigma}$ は近似的な対角行列である。
真の分解を $A = U \Sigma V^T$ とし、近似からの微小な修正量を以下のようにパラメータ化する。

$$
U = \tilde{U} (I + X), \quad V = \tilde{V} (I + Y), \quad \Sigma = \tilde{\Sigma} + \Delta \Sigma
$$

ここで、$X, Y \in \mathbb{R}^{n \times n}$ は**歪対称行列 (Skew-symmetric matrix)** ($X^T = -X, Y^T = -Y$) であり、直交性を保つための微小回転を表す。$\Delta \Sigma$ は対角行列である。

### 1.1 線形化された誤差方程式

$A = U \Sigma V^T$ に修正形式を代入し、2次の微小項 ($X \Delta \Sigma, X Y$ 等) を無視して線形化すると以下のようになる。

$$
\begin{aligned}
A &\approx \tilde{U} (I + X) (\tilde{\Sigma} + \Delta \Sigma) (I + Y)^T \tilde{V}^T \\
&\approx \tilde{U} (I + X) (\tilde{\Sigma} + \Delta \Sigma) (I - Y) \tilde{V}^T \quad (\because (I+Y)^T \approx I-Y \text{ for skew } Y) \\
&\approx \tilde{U} \tilde{\Sigma} \tilde{V}^T + \tilde{U} (X \tilde{\Sigma} - \tilde{\Sigma} Y + \Delta \Sigma) \tilde{V}^T
\end{aligned}
$$

ここで、残差行列 $R$ を $R = A - \tilde{U} \tilde{\Sigma} \tilde{V}^T$ と定義する。
上記の関係式より、以下の基本方程式が得られる。

$$
\tilde{U}^T R \tilde{V} = X \tilde{\Sigma} - \tilde{\Sigma} Y + \Delta \Sigma
$$

左辺を $F = \tilde{U}^T R \tilde{V}$ と置く。$F$ は現在の近似分解から計算可能な既知の行列である。
方程式は成分ごとに以下のように書ける ($1 \le i, j \le n$)。

$$
F_{ij} = X_{ij} \sigma_j - \sigma_i Y_{ij} + \Delta \sigma_{ij}
$$

ここで $\sigma_k = \tilde{\Sigma}_{kk}$、$\Delta \sigma_{ij}$ は $\Delta \Sigma$ の成分（$i \ne j$ なら0）である。

-----

## 2\. 方程式の解法

### 2.1 対角成分 ($i = j$) の決定

$i=j$ の場合、歪対称性より $X_{ii} = 0, Y_{ii} = 0$ であるため、特異値の修正量は直ちに求まる。

$$
\Delta \sigma_i = F_{ii}
$$

### 2.2 非対角成分 ($i \ne j$) の決定

$i \ne j$ の場合、$\Delta \Sigma_{ij} = 0$ である。
歪対称性 $X_{ji} = -X_{ij}, Y_{ji} = -Y_{ij}$ を利用すると、$(i, j)$ 成分と $(j, i)$ 成分の方程式は、$X_{ij}, Y_{ij}$ に関する $2 \times 2$ の連立一次方程式となる。

1.  $F_{ij} = \sigma_j X_{ij} - \sigma_i Y_{ij}$
2.  $F_{ji} = \sigma_i X_{ji} - \sigma_j Y_{ji} = -\sigma_i X_{ij} + \sigma_j Y_{ij}$

これを整理すると、以下の線形システム $M \mathbf{z} = \mathbf{b}$ を得る。

$$
\begin{pmatrix}
\sigma_j & -\sigma_i \\
-\sigma_i & \sigma_j
\end{pmatrix}
\begin{pmatrix}
X_{ij} \\
Y_{ij}
\end{pmatrix}
=
\begin{pmatrix}
F_{ij} \\
F_{ji}
\end{pmatrix}
$$

この行列 $M$ の行列式は $\det(M) = \sigma_j^2 - \sigma_i^2$ である。

-----

## 3\. 縮退・近接特異値の処理 (重要)

実装上の最大の課題は、特異値が近接している場合 ($\sigma_i \approx \sigma_j$) に $\det(M) \approx 0$ となり、上記システムが特異または悪条件になることである。Ogita らの論文では詳細が省略されているが、数学的に妥当な処理は**Moore-Penrose 一般化逆行列を用いた最小ノルム解**の採用である。

### 3.1 孤立している場合 ($\sigma_i \not\approx \sigma_j$)

通常通り Cramer の公式等で解く。

$$
X_{ij} = \frac{\sigma_j F_{ij} + \sigma_i F_{ji}}{\sigma_j^2 - \sigma_i^2}, \quad
Y_{ij} = \frac{\sigma_i F_{ij} + \sigma_j F_{ji}}{\sigma_j^2 - \sigma_i^2}
$$

### 3.2 縮退している場合 ($\sigma_i \approx \sigma_j$)

$|\sigma_i - \sigma_j| < \epsilon$（閾値は後述）の場合、行列 $M$ はランク落ち（ランク1）とみなせる。

$$
M \approx \sigma_i \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}
$$

このとき、方程式は $\sigma_i (X_{ij} - Y_{ij}) = F_{ij}$ かつ $-\sigma_i (X_{ij} - Y_{ij}) = F_{ji}$ となり、これは $F_{ij} \approx -F_{ji}$ であれば矛盾しない（$F$ の反対称成分が支配的になる）。
しかし、$X_{ij}$ と $Y_{ij}$ を個別に決定することはできず、自由度が残る。ここで**解のノルム $\sqrt{X_{ij}^2 + Y_{ij}^2}$ を最小化する解**を採用する。これは、近似部分空間内での不要な回転を抑える効果がある。

$2 \times 2$ 行列 $M$ の一般化逆行列 $M^\dagger$ を解析的に求めると、以下の計算式となる。

$$
\begin{pmatrix}
X_{ij} \\
Y_{ij}
\end{pmatrix}
=
M^\dagger \mathbf{b}
$$

縮退時（完全一致 $\sigma_i = \sigma_j = \sigma$ と仮定）の $M^\dagger$ は以下のようになる。

$$
M^\dagger = \frac{1}{4\sigma} \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}
$$

これを用いると、

$$
X_{ij} = \frac{F_{ij} - F_{ji}}{4\sigma}, \quad Y_{ij} = \frac{-F_{ij} + F_{ji}}{4\sigma} = -X_{ij}
$$

となる。これは直感的にも $X_{ij} = -Y_{ij}$ (左右の回転が逆方向に作用して打ち消し合う成分のみ抽出) という妥当な解である。

**実装上の注意:**
完全な縮退でなくとも、悪条件の場合は上記 $M^\dagger$ に近い挙動をさせるべきである。実装上は `numpy.linalg.pinv` をループ内で呼ぶと遅いため、以下の条件分岐付き明示公式を用いることを推奨する。

**【厳密な実装ロジック】**
各ペア $(i, j)$ ($i < j$) について、

1.  行列式 $D = \sigma_j^2 - \sigma_i^2$ を計算する。
2.  $|D| > \text{tol}$ の場合（通常ケース）:
      * 3.1節の公式で $X_{ij}, Y_{ij}$ を計算。
3.  $|D| \le \text{tol}$ の場合（近接ケース）:
      * $S = \sigma_i^2 + \sigma_j^2$ (これは $M^T M$ のトレースに関連) を計算。

      * $S \approx 0$ (両方ゼロに近い) ならば、$X_{ij} = Y_{ij} = 0$ (ノイズ増幅回避)。

      * それ以外 ($S > 0$) ならば、以下の近似式（最小二乗解）を使用:

        $$
        P = F_{ij} - F_{ji}
        $$

        $$
        X_{ij} = \frac{\sigma_j P}{S + \sigma_i \sigma_j \times 0}, \quad Y_{ij} = \frac{-\sigma_i P}{S}
        $$

        ※ 注: 完全縮退 $\sigma_i=\sigma_j$ のとき、これは $X_{ij} = \frac{P}{2\sigma}, Y_{ij} = \frac{-P}{2\sigma}$ となり、前述の $M^\dagger$ の結果（分母 $4\sigma$）と係数が異なるように見えるが、これは方程式の重み付け（$M$ の形）に依存する。
        最も安全かつ Ogita-Aishima の精神に沿った（特異部分空間を動かさない）解は、**縮退した特異値間の回転 $X_{ij}, Y_{ij}$ を強制的に 0 にする**ことである。

        **【推奨実装】**
        Ogita 論文の文脈（EVDにおけるクラスタ内回転の抑制）を SVD に適用すると：

          * $|\sigma_i - \sigma_j| / \max(1, |\sigma_i|, |\sigma_j|) < \text{tol}$ の場合：

            $$
            X_{ij} = 0, \quad Y_{ij} = 0
            $$

            とする。これにより、近接特異値に対応する特異ベクトルが不用意に混ざる（ドリフトする）のを防ぐ。これは数値積分法とのハイブリッド適用において、連続性を維持するために極めて重要である。

## 4\. 実装アルゴリズム詳細

### 入力

  * $A$: 対象行列 (時刻 $t$)
  * $U, S, V^T$: 現在の近似 SVD (時刻 $t-\Delta t$ の結果や予測値)
  * `tol`: 縮退判定の閾値 (例: $10^{-8}$ 〜 $10^{-12}$)

### 手順

1.  **残差計算**:

    $$
    R = A - (U \cdot \text{diag}(S) \cdot V^T)
    $$

    $$
    F = U^T \cdot R \cdot V
    $$

    ※ 計算コスト削減のため、完全に $A$ を再構築せずに $F = U^T A V - \text{diag}(S)$ と計算してもよい。数学的に等価である。

2.  **補正行列の構築**:
    $X, Y$ を $n \times n$ のゼロ行列、$\Delta S$ を長さ $n$ のゼロベクトルで初期化。

    `for i in range(n):`
    `for j in range(i, n):`
    `if i == j:`
    `dS[i] = F[i, i]`
    `else:`
    `# 3.1 & 3.2 のロジック`
    `det = S[j]**2 - S[i]**2`
    `if abs(det) > tol:`
    `X[i, j] = (S[j]*F[i, j] + S[i]*F[j, i]) / det`
    `Y[i, j] = (S[i]*F[i, j] + S[j]*F[j, i]) / det`
    `else:`
    `# 縮退時は回転させない (安定性重視)`
    `X[i, j] = 0.0`
    `Y[i, j] = 0.0`

    ```
            `# 歪対称性の適用`
            `X[j, i] = -X[i, j]`
            `Y[j, i] = -Y[i, j]`
    ```

3.  **更新**:
    本来は $U_{new} = U \exp(X), V_{new} = V \exp(Y)$ だが、1ステップの修正量が微小であることを前提に、以下の1次近似と再直交化を用いるのが計算コスト的に有利である。

    $$
    U_{tmp} = U (I + X), \quad V_{tmp} = V (I + Y)
    $$

    $$
    S_{new} = S + \Delta S
    $$

    最後に $U_{tmp}, V_{tmp}$ に対して QR分解等を用いて再直交化 (Re-orthogonalization) を行う。

    $$
    U_{new}, \_ = \text{qr}(U_{tmp})
    $$

    $$
    V_{new}, \_ = \text{qr}(V_{tmp})
    $$

### 出力

  * $U_{new}, S_{new}, V_{new}^T$

## 5\. 補足：既存コードとの統合について

`eigenpairflow` は時系列データを扱うため、前時刻からの連続性が重要である。
縮退時の処理で $X_{ij}=0$ とすることは、「解が一意に定まらない（特異ベクトルが回転してもSVDの誤差が変わらない）方向には、あえて動かさない」ことを意味し、これは**時間的な連続性を維持する上で最適な戦略**である。無理に最小ノルム解などを計算して微小なノイズに反応させるよりも、ゼロ固定の方がトラッキングの安定性は向上する。

