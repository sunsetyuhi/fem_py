# fem_py

Those are FEM (finite element method) programs written in Python 3. To run these programs, you also need to install SciPy and Matplotlib.

ここにあるのは、Python3で書いた有限要素法プログラムです。これらのプログラムを実行するには、SciPyとMatplotlibもインストールする必要があります。


# 解きたい問題（1次元問題の例）

　次式はPoisson（ポアソン）方程式と呼ばれる微分方程式で、熱伝導や物質拡散、静電場（静磁場）や<a href="https://teenaka.at.webry.info/201704/article_7.html">重力場</a>、引張圧縮応力、<a href="https://takun-physics.net/10186/">非圧縮性流れ</a>などを考える時に登場します。
 
$$ \frac{d}{dx} \left[ p(x) \frac{d u(x)}{dx} \right] = f(x)~~~ (a < x < b) $$

　また、次式はHelmholtz（ヘルムホルツ）方程式と呼ばれる微分方程式で、水面波、固体の振動、音波、電磁波などを考える時に登場します。

$$ \frac{d}{dx} \left[ p(x) \frac{d u(x)}{dx} \right] +q(x) u(x) = 0~~~ (a < x < b) $$
 
　プログラムでは、$x$は位置座標、$p(x),q(x),f(x)$は既知の関数とした際に、未知関数である$u(x)$を計算します。なお、微分方程式だけでは解が一意に定まらないため、プログラムでは何か適当な境界条件を定める必要があります。例えば、境界条件は次のように置きます。

$$ u(a) = \alpha,~~~     \frac{du(b)}{dx} = \beta $$

　上式の第1式は定義域の左端で$u(x)$の値を直接決めるDirichlet（ディリクレ）境界条件、第2式は定義域の右端で$u(x)$の傾きを決めるNeumann（ノイマン）境界条件です。このように、微分方程式と境界条件からなる問題は、「境界値問題」と呼ばれます。
 
　Pythonプログラムでは、メイン実行部で計算条件を設定しています。プログラムに興味がある方は、以上の内容を適宜参照していただければと思います。

