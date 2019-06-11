#1次元Helmholtz方程式を、有限要素法で解く
#常微分方程式： d/dx[p(x)du(x)/dx] +q(x)u(x) = 0  (x_min<x<x_max)
#境界条件： u(x_min)=alpha,  du(x_max)/dx=beta
import time  #時刻を扱うライブラリ
import numpy as np  #数値計算用
import scipy.linalg  #SciPyの線形計算ソルバー
import matplotlib.pyplot as plt  #グラフ作成


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
    mat_A_ele = np.zeros((len(nod_num_seg),2,2), np.float64)  #要素係数行列(ゼロで初期化)
    mat_B_ele = np.zeros((len(nod_num_seg),2,2), np.float64)  #要素係数行列(ゼロで初期化)

    #要素行列の各成分を計算
    print("Local matrix")
    for e in range(len(nod_num_seg)):
        for i in range(2):
            for j in range(2):
                mat_A_ele[e,i,j] = ( (-1)**(i+1) *(-1)**(j+1) ) / length[e]
                #mat_B_ele[e,i,j] = -func_f *length[e]/2.0

                if(i==j):
                    mat_B_ele[e,i,j] = cons_q*length[e]/3.0
                else:
                    mat_B_ele[e,i,j] = cons_q*length[e]/6.0

    return mat_A_ele, mat_B_ele


#全体方程式を構築
def assemble_global_matrix(mat_A_ele, mat_B_ele):
    #全体行列の初期化
    mat_A_glo = np.zeros((len(nod_pos_glo),len(nod_pos_glo)), np.float64) #全体係数行列(ゼロで初期化)
    mat_B_glo = np.zeros((len(nod_pos_glo),len(nod_pos_glo)), np.float64) #全体係数行列(ゼロで初期化)

    #要素行列から全体行列を組み立てる
    print("Global matrix (constructed)")
    for e in range(len(nod_num_seg)):
        for i in range(2):
            for j in range(2):
                mat_A_glo[ nod_num_seg[e,i], nod_num_seg[e,j] ] += mat_A_ele[e,i,j]
                mat_B_glo[ nod_num_seg[e,i], nod_num_seg[e,j] ] += mat_B_ele[e,i,j]

    print('Pre global matrix A')
    for i in range(min(len(nod_pos_glo),10)):   #全体行列を10行10列まで確認
        for j in range(min(len(nod_pos_glo),10)):
            print("{:7.2f}".format(mat_A_glo[i,j]), end='')
        print()
    print('Pre global matrix B')
    for i in range(min(len(nod_pos_glo),10)):   #全体行列を10行10列まで確認
        for j in range(min(len(nod_pos_glo),10)):
            print("{:7.2f}".format(mat_B_glo[i,j]), end='')
        print()

    return mat_A_glo, mat_B_glo


#境界要素の情報を設定
def make_boundary_info(nod_pos_seg):
    BC_type = [""]*2
    BC_value = [""]*2

    #左側境界
    BC_type[0] = BC_left[0]
    BC_value[0] = BC_left[1]

    #右側境界
    BC_type[1] = BC_right[0]
    BC_value[1] = BC_right[1]

    print('BC_type =\n', BC_type)

    return BC_type, BC_value


#境界条件を実装
def set_boundary_condition(mat_A_glo, mat_B_glo, BC_type, BC_value):
    BC_nod = [0,len(nod_pos_glo)-1]

    #各要素の各節点に対応したGlobal節点に対して処理する
    print('Boundary conditions')
    for n in range(2):
        if(BC_type[n]=='Dirichlet'):
            mat_A_glo[BC_nod[n], :] = 0.0  #行を全て0にする
            mat_A_glo[:, BC_nod[n]] = 0.0  #列を全て0にする
            mat_A_glo[BC_nod[n], BC_nod[n]] = 1.0  #対角成分は1にする

    print('Post global matrix A')
    for i in range(min(len(nod_pos_glo),10)):   #全体行列を10行10列まで確認
        for j in range(min(len(nod_pos_glo),10)):
            print("{:7.2f}".format(mat_A_glo[i,j]), end='')
        print()
    print('Post global matrix B')
    for i in range(min(len(nod_pos_glo),10)):   #全体行列を10行10列まで確認
        for j in range(min(len(nod_pos_glo),10)):
            print("{:7.2f}".format(mat_B_glo[i,j]), end='')
        print()

    return mat_A_glo, mat_B_glo


#連立方程式を解く
def solve_simultaneous_equations(mat_A_glo, mat_B_glo):
    print('節点数、境界線分要素数')
    print(len(nod_pos_glo), len(nod_pos_seg))

    print('Solve linear equations')
    #Au=λBuから、固有値Eigと固有値ベクトルUを求める
    eigenvalues, unknown_vec_u = scipy.linalg.eigh(mat_A_glo, mat_B_glo)

    print("Eigenvalues =\n", eigenvalues)  #固有値
    print("Unkown vector U =\n", unknown_vec_u)  #未知数ベクトル

    print("N_Eig = ", np.count_nonzero(eigenvalues))  #非ゼロの固有値の個数
    eigenvalues_nonzero = eigenvalues[np.where(0.000001<abs(eigenvalues))]
    print("eigenvalues_nonzero =\n", eigenvalues_nonzero)  #固有値の非ゼロ成分

    return unknown_vec_u, eigenvalues


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
def visualize_result(nod_pos_glo, unknown_vec_u, show_text=False, out_type='show'):
    plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
    fig = plt.figure(figsize=(16, 12), dpi=100, facecolor='#ffffff')

    fig.suptitle("FEA of 1D Helmholtz's equation", fontsize=16)  #全体のグラフタイトル

    #表示する固有値・固有ベクトルの番号
    if (BC_left[0]=='Dirichlet' or BC_right[0]=='Dirichlet'):
        count = 0
    else:
        count = 1

    #数値解をプロット
    for i in range(plot_num[0]*plot_num[1]):
        ax = fig.add_subplot(plot_num[0], plot_num[1], i+1)
        #ax.set_title("Eig={:0.5f}".format(eigenvalues[count]))
        ax.set_title("k={:0.5f}".format( np.sqrt(eigenvalues[count]) ))
        ax.plot(nod_pos_glo,unknown_vec_u[:,count], color='#0000ff')  #折線グラフを作成
        count += 1
    plt.tight_layout()  #余白を調整
    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.9, wspace=0.1, hspace=0.3)  #余白を調整

    #グラフを表示
    if(out_type=='show'):
        plt.show()
    elif(out_type=='save'):
        plt.savefig("fem1d_helmholtz.png")
    plt.close()  #作成した図のウィンドウを消す


#メイン実行部
if __name__ == '__main__':
    ##### プリプロセス #####
    x_min = -1.0  #計算領域のXの最小値
    x_max = 1.0  #計算領域のXの最大値
    cons_p = 1.0  #定数項p
    cons_q = 1.0  #定数項q
    omega = 1.0  #k0=ω/c、カットオフ波数(ωa/2πc=a/λ)

    #左部(x=x_min)、右部(x=x_max)の、境界の種類と値
    #境界の種類はNone,Dirichlet,Neumann
    BC_left = ['Dirichlet', 0.0]
    BC_right = ['Dirichlet', 0.0]

    #BC_left = ['Neumann', 0.0]
    #BC_right = ['Neumann', 0.0]

    node_type = ['lattice', 500]  #格子点配置、格子分割におけるx・y方向の節点数
    #node_type = ['random', 500]  #ランダム配置、ランダム分割における節点数

    #節点データ生成。Global節点座標、線分要素の節点番号
    nod_pos_glo, nod_num_seg = generate_nodes(node_type)

    #Local節点座標を作成。線分要素のLocal節点座標
    nod_pos_seg = make_mesh_data()

    #メッシュを表示(ポストプロセス)。番号などの有無(True,False)、グラフの表示方法(show,save)
    visualize_mesh(nod_pos_glo, show_text=False, out_type='show')


    ##### メインプロセス #####
    #計算の開始時刻を記録
    print ("Calculation start: ", time.ctime())  #計算開始時刻を表示
    compute_time = time.time()  #計算の開始時刻

    #要素方程式の構築
    mat_A_ele, mat_B_ele = assemble_element_matrix(nod_num_seg, nod_pos_seg)

    #全体方程式の構築
    mat_A_glo, mat_B_glo = assemble_global_matrix(mat_A_ele, mat_B_ele)

    #境界要素の情報を設定
    BC_type, BC_value = make_boundary_info(nod_pos_seg)

    #境界条件を実装
    mat_A_glo, mat_B_glo = set_boundary_condition(mat_A_glo, mat_B_glo, BC_type, BC_value)

    #連立方程式を解く
    unknown_vec_u, eigenvalues = solve_simultaneous_equations(mat_A_glo, mat_B_glo)

    #計算時間の表示
    compute_time = time.time() -compute_time
    print ("Calculation time: {:0.5f}[sec]".format(compute_time))

    #計算結果を表示(ポストプロセス)。番号などの有無(True,False)、グラフの表示方法(show,save)
    plot_num = [3, 4]  #グラフの縦横の作成数
    visualize_result(nod_pos_glo, unknown_vec_u, show_text=False, out_type='show')
