#1次元Poisson方程式を、有限要素法で解く
#常微分方程式： d/dx[p(x)du(x)/dx] = f(x)  (x_min<x<x_max)
#境界条件： u(x_min)=alpha,  du(x_max)/dx=beta
import time  #時刻を扱うライブラリ
import numpy as np  #NumPyライブラリ
import scipy.linalg  #SciPyの線形計算ライブラリ
import matplotlib.pyplot as plt  #データ可視化ライブラリ


#節点データを生成
def generate_nodes(node_type):
    node_total = node_type[1]  #節点数(>=2)
    ele_total = node_total-1  #要素数

    #格子点配置
    if (node_type[0]=='lattice'):
        #Global節点のx座標を定義(x_min〜x_max)
        lattice_num = node_type[1]  #格子点分割における節点数
        nod_pos_glo = np.linspace(x_min,x_max, lattice_num) #計算領域を等分割

    #ランダム配置
    elif (node_type[0]=='random'):
        random_num = node_type[1]  #ランダム分割における節点数
        nod_pos_glo = np.random.rand(random_num)  #[0~1]の点をrandom_num個生成
        nod_pos_glo = np.sort(nod_pos_glo)  #昇順(小さい順)にソート
        nod_pos_glo = x_min +(x_max-x_min)*nod_pos_glo

        #隅に点を移動
        if (2<=random_num):
            nod_pos_glo[0] = x_min
            nod_pos_glo[node_total-1] = x_max

    print('Global節点のx,y座標\n', nod_pos_glo)

    #各線分要素のGlobal節点番号
    nod_num_seg = np.empty((ele_total,2), np.int)
    for e in range(ele_total):
        nod_num_seg[e,0] = e
        nod_num_seg[e,1] = e+1
    print('線分要素を構成するGlobal節点番号\n', nod_num_seg)

    return nod_pos_glo, nod_num_seg


#入力データの用意
def make_mesh_data():
    #print("node_total = ",node_total, ",  ele_total = ",ele_total)

    #各線分要素のLocal節点のx座標
    print('線分要素を構成するLocal節点座標')
    nod_pos_seg = np.empty((len(nod_num_seg),2), np.float64)
    for e in range(len(nod_num_seg)):
        for n in range(2):
            nod_pos_seg[e,n] = nod_pos_glo[ nod_num_seg[e,n] ]
    print(nod_pos_seg)

    return nod_pos_seg


#要素方程式を構築
def assemble_element_matrix(nod_num_seg, nod_pos_seg):
    #各線分要素の長さ
    print("Element length")
    length = np.empty(len(nod_num_seg), np.float64)
    for e in range(len(nod_num_seg)):
        length[e] = np.absolute( nod_pos_seg[e,1] -nod_pos_seg[e,0] )
    print(length)

    #要素行列の初期化
    mat_A_ele = np.zeros((len(nod_num_seg),3,3), np.float64)  #要素係数行列(ゼロで初期化)
    vec_b_ele = np.zeros((len(nod_num_seg),3), np.float64)  #要素係数ベクトル(ゼロで初期化)

    #要素行列の各成分を計算
    print("Local matrix")
    for e in range(len(nod_num_seg)):
        for i in range(2):
            for j in range(2):
                mat_A_ele[e,i,j] = ( (-1)**(i+1) *(-1)**(j+1) ) / length[e]
            vec_b_ele[e,i] = -func_f *length[e]/2.0

    return mat_A_ele, vec_b_ele


#全体方程式を構築
def assemble_global_matrix(mat_A_ele, vec_b_ele):
    #全体行列の初期化
    mat_A_glo = np.zeros((len(nod_pos_glo),len(nod_pos_glo)), np.float64) #全体係数行列(ゼロで初期化)
    vec_b_glo = np.zeros(len(nod_pos_glo), np.float64) #全体係数ベクトル(ゼロで初期化)

    #要素行列から全体行列を組み立てる
    print("Global matrix (constructed)")
    for e in range(len(nod_num_seg)):
        for i in range(2):
            for j in range(2):
                mat_A_glo[ nod_num_seg[e,i], nod_num_seg[e,j] ] += mat_A_ele[e,i,j]
            vec_b_glo[ nod_num_seg[e,i] ] += vec_b_ele[e,i]
    print(np.concatenate((mat_A_glo, np.reshape(vec_b_glo, (-1,1))), axis=1))

    return mat_A_glo, vec_b_glo


def set_boundary_condition(mat_A_glo, vec_b_glo):
    print("Boundary conditions")
    #boundary(mat_A_glo, vec_b_glo, 0, alpha, 0.0) #左端はディリクレ境界
    #boundary(mat_A_glo, vec_b_glo, node_total-1, "inf", beta) #右端はノイマン境界

    #ディリクレ境界条件
    vec_b_glo[:] -= BC_left[1]*mat_A_glo[0,:]  #定数ベクトルに行の値を移項
    vec_b_glo[0] = BC_left[1]  #関数を任意の値で固定
    mat_A_glo[0,:] = 0.0  #行を全て0にする
    mat_A_glo[:,0] = 0.0  #列を全て0にする
    mat_A_glo[0,0] = 1.0  #対角成分は1にする

    #ノイマン境界条件
    vec_b_glo[len(nod_pos_glo)-1] += BC_right[1] #関数を任意の傾きで固定

    print("Post global matrix")
    print(np.concatenate((mat_A_glo, np.reshape(vec_b_glo, (-1,1))), axis=1))

    return mat_A_glo, vec_b_glo


#連立方程式を解く
def solve_simultaneous_equations(mat_A_glo, vec_b_glo):
    print('節点数、境界線分要素数')
    print(len(nod_pos_glo), len(nod_pos_seg))
    #print('detA = ', scipy.linalg.det(mat_A_glo))  #Aの行列式
    #print('Rank A = ', np.linalg.matrix_rank(mat_A_glo))  #AのRank(階数)
    #print('Inverse A = ', scipy.linalg.inv(mat_A_glo))  #Aの逆行列

    #未知数ベクトル
    print('Unkown vector u = ')
    unknown_vec_u = scipy.linalg.solve(mat_A_glo,vec_b_glo)  #Au=bから、未知数ベクトルuを求める
    print(unknown_vec_u)
    print('Max u = ', max(unknown_vec_u), ',  Min u = ',min(unknown_vec_u))  #uの最大値、最小値

    return unknown_vec_u


#メッシュを表示
def visualize_mesh(nod_pos_glo, show_text, out_type):
    #plt.rcParams['font.family'] = 'Times New Roman'  #全体のフォントを設定
    fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')  #図の設定

    #plt.title("Finite element analysis of 1D Poisson's equation")  #グラフタイトル
    plt.xlabel("$x$")  #x軸の名前
    plt.ylabel("$u(x)$")  #y軸の名前
    plt.grid()  #点線の目盛りを表示

    #メッシュをプロット
    zero_list = np.zeros(len(nod_pos_glo))
    plt.plot(nod_pos_glo,zero_list, label="$\phi(x)$", color='#0000ff')  #線分要素
    plt.scatter(nod_pos_glo,zero_list, color='#0000ff')  #節点

    if(show_text==True):
        for n in range(len(nod_pos_glo)):  #節点番号
            plt.text(nod_pos_glo[n],zero_list[n], 'n%d' %n, ha='center',va='bottom', color='#0000ff')
        for e in range(len(nod_pos_seg)):  #線分要素番号
            meanX = (nod_pos_seg[e,0] +nod_pos_seg[e,1])/2.0
            plt.text(meanX, zero_list[n], 'e%d' %e, ha='center',va='top')

    #グラフを表示
    if(out_type=='show'):
        plt.show()
    elif(out_type=='save'):
        plt.savefig("fem1d_mesh.png")
        #plt.savefig('fem1d_mesh.pdf')
    plt.close()  #作成した図のウィンドウを消す


#計算結果を表示
def visualize_result(nod_pos_glo, unknown_vec_u, show_text, out_type):
    #plt.rcParams['font.family'] = 'Times New Roman'  #全体のフォントを設定
    fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')  #図の設定

    #plt.title("Finite element analysis of 1D Poisson's equation")  #グラフタイトル
    plt.xlabel("$x$")  #x軸の名前
    plt.ylabel("$u(x)$")  #y軸の名前
    plt.grid()  #点線の目盛りを表示

    #解析解をプロット
    if (BC_left[0]=='Dirichlet' and BC_right[0]=='Neumann'):
        exact_x = np.arange(x_min,x_max,0.01)
        exact_y = (func_f/2)*exact_x**2 +(-func_f*x_max +BC_right[1])*exact_x \
                 -(func_f/2)*x_min**2 -(-func_f*x_max +BC_right[1])*x_min +BC_left[1]
        plt.plot(exact_x,exact_y, label="$u(x)$", color='#ff0000')  #折線グラフを作成

    #数値解をプロット
    plt.plot(nod_pos_glo,unknown_vec_u, label="$\hat{u}(x)$", color='#0000ff')  #折線グラフを作成
    plt.scatter(nod_pos_glo,unknown_vec_u)  #点グラフを作成

    #更に体裁を整える
    plt.legend(loc='best')  #凡例(グラフラベル)を表示

    if(show_text==True):
        for n in range(len(nod_pos_glo)):  #節点番号
            plt.text(nod_pos_glo[n],unknown_vec_u[n], 'n%d' %n, ha='center',va='bottom', color='#0000ff')
        for e in range(len(nod_pos_seg)):  #線分要素番号
            meanX = (nod_pos_seg[e,0] +nod_pos_seg[e,1])/2.0
            meanU = (unknown_vec_u[nod_num_seg[e,0]] +unknown_vec_u[nod_num_seg[e,1]])/2.0
            plt.text(meanX, meanU, 'e%d' %e, ha='center',va='top')

    #グラフを表示
    if(out_type=='show'):
        plt.show()
    elif(out_type=='save'):
        plt.savefig("fem1d_poisson.png")
        #plt.savefig('fem1d_poisson.pdf')
    plt.close()  #作成した図のウィンドウを消す


#メイン実行部
if __name__ == '__main__':
    ##### プリプロセス #####
    x_min = -1.0  #計算領域のXの最小値
    x_max = 1.0  #計算領域のXの最大値
    func_f = 1.0  #定数関数f

    #左部(x=x_min)、右部(x=x_max)の、境界の種類と値
    #境界の種類はNone,Dirichlet,Neumann
    BC_left = ['Dirichlet', 0.0]
    BC_right = ['Neumann', 1.0]

    #node_type = ['lattice', 10]  #格子点配置、格子分割におけるx・y方向の節点数
    node_type = ['random', 10]  #ランダム配置、ランダム分割における節点数

    #節点データ生成。Global節点座標、線分要素の節点番号
    nod_pos_glo, nod_num_seg = generate_nodes(node_type)

    #Local節点座標を作成。線分要素のLocal節点座標
    nod_pos_seg = make_mesh_data()

    #メッシュを表示(ポストプロセス)。番号などの有無(True,False)、グラフの表示方法(show,save)
    visualize_mesh(nod_pos_glo, show_text=True, out_type='show')


    ##### メインプロセス #####
    #計算の開始時刻を記録
    print ("Calculation start: ", time.ctime())  #計算開始時刻を表示
    compute_time = time.time()  #計算の開始時刻

    #要素方程式の構築
    mat_A_ele, vec_b_ele = assemble_element_matrix(nod_num_seg, nod_pos_seg)

    #全体方程式の構築
    mat_A_glo, vec_b_glo = assemble_global_matrix(mat_A_ele, vec_b_ele)

    #境界条件を実装
    mat_A_glo, vec_b_glo = set_boundary_condition(mat_A_glo, vec_b_glo)

    #連立方程式を解く
    unknown_vec_u = solve_simultaneous_equations(mat_A_glo, vec_b_glo)

    #計算時間の表示
    compute_time = time.time() -compute_time
    print ("Calculation time: {:0.5f}[sec]".format(compute_time))

    #計算結果を表示(ポストプロセス)。番号などの有無(True,False)、グラフの表示方法(show,save)
    visualize_result(nod_pos_glo, unknown_vec_u, show_text=True, out_type='show')
