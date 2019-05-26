#2次元Poisson方程式を、有限要素法で解く
#偏微分方程式： -∇・[p(x,y)∇u(x,y)] = f(x,y)  (in Ω)
#境界条件： u(x,y)=alpha  (in Γ1),  du(x,y)/dx=beta  (in Γ2)
import time  #時刻を扱うライブラリ
import numpy as np  #数値計算用
import scipy.spatial  #ドロネー分割
import scipy.linalg  #SciPyの線形計算ソルバー
import scipy.sparse  #圧縮行列の処理
import scipy.sparse.linalg  #圧縮行列用ソルバー
import matplotlib.pyplot as plt  #グラフ作成
from mpl_toolkits.mplot3d import Axes3D  #3Dグラフ
from matplotlib import cm  #カラーマップ


#節点データを生成
def generate_nodes(node_type):
    #格子点配置
    if (node_type[0]=='lattice'):
        lattice_num = node_type[1]  #格子分割におけるx・y方向の節点数
        x = np.linspace(x_min, x_max, lattice_num)
        y = np.linspace(y_min, y_max, lattice_num)

        nodes = np.empty((lattice_num*lattice_num,2), np.float64)
        for j in range(lattice_num):
            for i in range(lattice_num):
                nodes[i +lattice_num*j, 0] = x[i]
                nodes[i +lattice_num*j, 1] = y[j]

    #ランダム配置
    elif (node_type[0]=='random'):
        random_num = node_type[1]  #ランダム分割における節点数
        nodes = np.random.rand(random_num,2)  #[0~1,0~1]の点をrandom_num個生成
        nodes[:,0] = x_min +(x_max-x_min)*nodes[:,0]
        nodes[:,1] = y_min +(y_max-y_min)*nodes[:,1]

        #隅に点を移動
        if (4<=random_num):
            nodes[0,0],nodes[0,1] = (x_min, (y_max+y_min)/2)
            nodes[1,0],nodes[1,1] = (x_max, (y_max+y_min)/2)
            nodes[2,0],nodes[2,1] = ((x_max+x_min)/2, y_min)
            nodes[3,0],nodes[3,1] = ((x_max+x_min)/2, y_max)
            '''nodes[4,0],nodes[4,1] = (x_min, y_min)
            nodes[5,0],nodes[5,1] = (x_min, y_max)
            nodes[6,0],nodes[6,1] = (x_max, y_min)
            nodes[7,0],nodes[7,1] = (x_max, y_max)  #'''

    #節点をドロネー分割
    delaunay_data = scipy.spatial.Delaunay(nodes)

    nod_total = delaunay_data.points.shape[0]
    tri_ele_total = delaunay_data.simplices.shape[0]
    seg_ele_total = delaunay_data.convex_hull.shape[0]
    print('節点数、三角形要素数、境界線分要素数')
    print(nod_total, tri_ele_total, seg_ele_total)

    nod_pos_glo = delaunay_data.points  #[nod_total,2]
    print('Global節点のx,y座標\n', nod_pos_glo)

    nod_num_tri = delaunay_data.simplices  #[tri_ele_total,3]
    print('三角形要素を構成する節点番号\n', nod_num_tri)

    nod_num_seg = delaunay_data.convex_hull  #[seg_ele_total,2]
    print('境界線分要素を構成する節点番号\n', nod_num_seg)

    return nod_pos_glo, nod_num_tri, nod_num_seg


def make_mesh_data():
    print('三角形要素を構成するLocal節点座標')
    nod_pos_tri = np.empty((len(nod_num_tri),3,2), np.float64)  #各要素のLocal節点のx,y座標
    for e in range(len(nod_num_tri)):
        for n in range(3):
            nod_pos_tri[e,n,0] = nod_pos_glo[nod_num_tri[e,n], 0]
            nod_pos_tri[e,n,1] = nod_pos_glo[nod_num_tri[e,n], 1]
    print('nod_pos_tri(x0, y0),(x1, y1),(x2, y2) =\n', nod_pos_tri)

    print('境界線分要素を構成するLocal節点座標')
    nod_pos_seg = np.empty((len(nod_num_seg),2,2), np.float64)  #各境界要素のLocal節点のx,y座標
    for e in range(len(nod_num_seg)):
        for n in range(2):
            nod_pos_seg[e,n,0] = nod_pos_glo[nod_num_seg[e,n], 0]
            nod_pos_seg[e,n,1] = nod_pos_glo[nod_num_seg[e,n], 1]
    print('nod_pos_seg(x0, y0),(x1, y1) =\n', nod_pos_seg)

    return nod_pos_tri, nod_pos_seg


#要素行列の構築
def assemble_element_matrix(nod_num_tri, nod_pos_tri):
    #各要素の面積
    print('Element area_tri')
    area_tri = (nod_pos_tri[:,1,0]-nod_pos_tri[:,0,0])*(nod_pos_tri[:,2,1]-nod_pos_tri[:,0,1])  \
              -(nod_pos_tri[:,2,0]-nod_pos_tri[:,0,0])*(nod_pos_tri[:,1,1]-nod_pos_tri[:,0,1])
    area_tri = np.absolute(area_tri)/2.0
    print(area_tri)

    #各要素の形状関数の係数
    print('Shape function a,b,c')
    shape_a = np.empty((len(nod_pos_tri),3), np.float64)
    shape_b = np.empty((len(nod_pos_tri),3), np.float64)
    shape_c = np.empty((len(nod_pos_tri),3), np.float64)
    for e in range(len(nod_pos_tri)):
        shape_a[e,0] = nod_pos_tri[e,1,0]*nod_pos_tri[e,2,1] -nod_pos_tri[e,2,0]*nod_pos_tri[e,1,1]
        shape_a[e,1] = nod_pos_tri[e,2,0]*nod_pos_tri[e,0,1] -nod_pos_tri[e,0,0]*nod_pos_tri[e,2,1]
        shape_a[e,2] = nod_pos_tri[e,0,0]*nod_pos_tri[e,1,1] -nod_pos_tri[e,1,0]*nod_pos_tri[e,0,1]

        shape_b[e,0] = nod_pos_tri[e,1,1] -nod_pos_tri[e,2,1]
        shape_b[e,1] = nod_pos_tri[e,2,1] -nod_pos_tri[e,0,1]
        shape_b[e,2] = nod_pos_tri[e,0,1] -nod_pos_tri[e,1,1]

        shape_c[e,0] = nod_pos_tri[e,2,0] -nod_pos_tri[e,1,0]
        shape_c[e,1] = nod_pos_tri[e,0,0] -nod_pos_tri[e,2,0]
        shape_c[e,2] = nod_pos_tri[e,1,0] -nod_pos_tri[e,0,0]
    for e in range(min(len(nod_pos_tri),10)):  #形状関数の係数を10番目の三角形要素まで確認
        print(shape_a[e,:], shape_b[e,:], shape_c[e,:])

    #要素行列の初期化
    mat_A_ele = np.zeros((len(nod_num_tri),3,3), np.float64)  #要素係数行列(ゼロで初期化)
    vec_b_ele = np.zeros((len(nod_num_tri),3), np.float64)  #要素係数ベクトル(ゼロで初期化)

    print("Local matrix")
    for e in range(len(nod_pos_tri)):
        for i in range(3):
            for j in range(3):
                mat_A_ele[e,i,j] = (shape_b[e,i]*shape_b[e,j] +shape_c[e,i]*shape_c[e,j]) / (4.0*area_tri[e])
            vec_b_ele[e,i] = -func_f *area_tri[e]/3.0

    return mat_A_ele, vec_b_ele, area_tri


#全体行列の構築
def assemble_global_matrix(matrix_type):
    #全体行列を用意
    if(matrix_type=='basic'):
        mat_A_glo = np.zeros((len(nod_pos_glo),len(nod_pos_glo)), np.float64) #全体行列(ゼロで初期化)
    elif(matrix_type=='sparse'):
        mat_A_glo = scipy.sparse.lil_matrix((len(nod_pos_glo),len(nod_pos_glo))) #lil形式の圧縮行列(ゼロで初期化)
    vec_b_glo = np.zeros(len(nod_pos_glo), np.float64) #全体ベクトル(ゼロで初期化)

    #全体行列を組み立てる
    print('Assemble matrix')
    CountPercent = 1
    for e in range(len(nod_pos_tri)):
        for i in range(3):
            for j in range(3):
                mat_A_glo[ nod_num_tri[e,i], nod_num_tri[e,j] ] += mat_A_ele[e,i,j]
            vec_b_glo[ nod_num_tri[e,i] ] += vec_b_ele[e,i]

        #処理の経過を％表示
        if(CountPercent <= 100*e/len(nod_pos_tri)):
            print("{:7.2f}%".format(100*e/len(nod_pos_tri)), end='')
            CountPercent += 1
    print(" 100.00%")

    print('Pre global matrix')
    for i in range(min(len(nod_pos_glo),10)):   #全体行列を10行10列まで確認
        for j in range(min(len(nod_pos_glo),10)):
            print("{:7.2f}".format(mat_A_glo[i,j]), end='')
        print(";{:7.2f}".format(vec_b_glo[i]))
    #print(np.concatenate((mat_A_glo, np.reshape(vec_b_glo, (-1,1))), axis=1))

    return mat_A_glo, vec_b_glo


#境界要素の情報を設定
def make_boundary_info(nod_pos_seg):
    #境界線分要素の長さ
    leng_seg = (nod_pos_seg[:,0,0]-nod_pos_seg[:,1,0])**2.0 +(nod_pos_seg[:,0,1]-nod_pos_seg[:,1,1])**2.0
    leng_seg = np.sqrt(leng_seg)
    print('leng_seg =\n', leng_seg)

    #境界要素の種類を分類
    BC_type = [""]*len(nod_pos_seg)
    BC_value = [""]*len(nod_pos_seg)
    BC_threshold = 0.5*np.mean(leng_seg)
    for e in range(len(nod_pos_seg)):
        if(nod_pos_seg[e,0,0] < x_min +BC_threshold and
           nod_pos_seg[e,1,0] < x_min +BC_threshold):  #左側境界
            BC_type[e] = BC_left[0]
            BC_value[e] = BC_left[1]
        elif(x_max -BC_threshold < nod_pos_seg[e,0,0] and
             x_max -BC_threshold < nod_pos_seg[e,1,0]):  #右側境界
            BC_type[e] = BC_right[0]
            BC_value[e] = BC_right[1]
        elif(nod_pos_seg[e,0,1] < y_min +BC_threshold and
             nod_pos_seg[e,1,1] < y_min +BC_threshold):  #下側境界
            BC_type[e] = BC_bottom[0]
            BC_value[e] = BC_bottom[1]
        elif(y_max -BC_threshold < nod_pos_seg[e,0,1] and
             y_max -BC_threshold < nod_pos_seg[e,1,1]):  #上側境界
            BC_type[e] = BC_top[0]
            BC_value[e] = BC_top[1]
        else:  #それ以外はNeumann境界にしておく（何もしない）
            BC_type[e] = 'Neumann'
            BC_value[e] = 0.0
    print('BC_type =\n', BC_type)
    print('BC_value =\n', BC_value)

    return BC_type, BC_value, leng_seg


#境界条件を実装
def set_boundary_condition(mat_A_glo, vec_b_glo, BC_type, BC_value, leng_seg):
    #各要素の各節点に対応したGlobal節点に対して処理する
    print('Boundary conditions')
    CountPercent = 1
    for e in range(len(nod_pos_seg)):
        for n in range(2):
            if(BC_type[e]=='Dirichlet'):
                vec_b_glo[:] -= BC_value[e]*mat_A_glo[nod_num_seg[e,n],:]  #移項
                vec_b_glo[nod_num_seg[e,n]] = BC_value[e]  #関数を任意の値で固定
                mat_A_glo[nod_num_seg[e,n],:] = 0.0  #行を全て0にする
                mat_A_glo[:,nod_num_seg[e,n]] = 0.0  #列を全て0にする
                mat_A_glo[nod_num_seg[e,n],nod_num_seg[e,n]] = 1.0  #対角成分は1にする

            if (BC_type[e]=='Neumann'):  #Neumann境界条件の処理
                vec_b_glo[nod_num_seg[e,n]] += BC_value[e]*leng_seg[e]/2.0  #関数を任意の傾きで固定

        #処理の経過を％表示
        if(CountPercent <= 100*e/len(nod_pos_seg)):
            print("{:7.2f}%".format(100*e/len(nod_pos_seg)), end='')
            CountPercent += 1
    print(" 100.00%")

    print("Post global matrix")
    for i in range(min(len(nod_pos_glo),10)):   #全体行列を10行10列まで確認
        for j in range(min(len(nod_pos_glo),10)):
            print("{:7.2f}".format(mat_A_glo[i,j]), end='')
        print(";{:7.2f}".format(vec_b_glo[i]))
    #print(np.concatenate((mat_A_glo, np.reshape(vec_b_glo, (-1,1))), axis=1))

    return mat_A_glo, vec_b_glo


#連立方程式を解く
def solve_simultaneous_equations(mat_A_glo, vec_b_glo):
    print('節点数、三角形要素数、境界線分要素数')
    print(len(nod_pos_glo), len(nod_pos_tri), len(nod_pos_seg))
    #print("detA = ", scipy.linalg.det(mat_A_glo)) #Aの行列式
    #print("Rank A = ", np.linalg.matrix_rank(mat_A_glo)) #AのRank(階数)
    #print("Inverse A = ", scipy.linalg.inv(mat_A_glo)) #Aの逆行列

    print('Solve linear equations')
    if(matrix_type=='basic'):
        unknown_vec_u = scipy.linalg.solve(mat_A_glo,vec_b_glo)  #Au=bから、未知数ベクトルUを求める
    elif(matrix_type=='sparse'):
        unknown_vec_u = scipy.sparse.linalg.spsolve(mat_A_glo,vec_b_glo)  #lil形式をcsr形式に変換して計算

    print('Unkown vector U = ') #未知数ベクトル
    print(unknown_vec_u)
    print('Max U = ', max(unknown_vec_u), ',  Min U = ',min(unknown_vec_u)) #uの最大値、最小値

    return unknown_vec_u


#メッシュを表示
def visualize_mesh(nod_pos_glo, show_text, out_type):
    #plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
    fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')  #図の設定

    #plt.title("2D mesh for FEM") #グラフタイトル
    plt.xlabel('$x$') #x軸の名前
    plt.ylabel('$y$') #y軸の名前

    #メッシュをプロット
    plt.triplot(nod_pos_glo[:,0],nod_pos_glo[:,1], nod_num_tri, color='#0000ff')  #三角形要素
    plt.scatter(nod_pos_glo[:,0],nod_pos_glo[:,1], color='#0000ff')  #節点

    if(show_text==True):
        for n in range(len(nod_pos_glo)):  #節点番号
            plt.text(nod_pos_glo[n,0], nod_pos_glo[n,1], n, ha='right')
        for e in range(len(nod_pos_tri)):  #三角形要素番号
            meanX = (nod_pos_tri[e,0,0] +nod_pos_tri[e,1,0] +nod_pos_tri[e,2,0])/3.0
            meanY = (nod_pos_tri[e,0,1] +nod_pos_tri[e,1,1] +nod_pos_tri[e,2,1])/3.0
            plt.text(meanX, meanY, '#%d' %e, ha='center')
        for e in range(len(nod_pos_seg)):  #線分要素番号
            meanX = (nod_pos_seg[e,0,0] +nod_pos_seg[e,1,0])/2.0
            meanY = (nod_pos_seg[e,0,1] +nod_pos_seg[e,1,1])/2.0
            plt.text(meanX, meanY, '*%d' %e, ha='center')

    #グラフを表示
    if(out_type=='show'):
        plt.show()
    elif(out_type=='save'):
        plt.savefig("fem2d_mesh.png")
    plt.close()  #作成した図のウィンドウを消す


#計算結果を表示
def visualize_result(nod_pos_glo, unknown_vec_u, show_text, out_type):
    #plt.rcParams['font.family'] = 'Times New Roman'  #全体のフォントを設定
    fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')  #図の設定
    ax = fig.gca(projection='3d', azim=-120, elev=20)  #3Dグラフを設定

    #plt.title("FEA of 2D Poisson's equation")  #グラフタイトル
    ax.set_xlabel('$x$')  #x軸の名前
    ax.set_ylabel('$y$')  #y軸の名前
    ax.set_zlabel('$u(x,y)$')  #z軸の名前

    #数値解をプロット
    surf = ax.plot_trisurf(nod_pos_glo[:,0],nod_pos_glo[:,1],unknown_vec_u, cmap=cm.jet, linewidth=0)
    plt.colorbar(surf, shrink=0.8, aspect=10)  #カラーバー

    #更に体裁を整える
    plt.legend(loc='best')  #凡例(グラフラベル)を表示

    if(show_text==True):
        for n in range(len(nod_pos_glo)):  #節点番号
            ax.text(nod_pos_glo[n,0],nod_pos_glo[n,1],unknown_vec_u[n], 'n%d' %n, ha='center',va='bottom', color='#000000')
        for e in range(len(nod_pos_tri)):  #三角形要素番号
            meanX = (nod_pos_tri[e,0,0] +nod_pos_tri[e,1,0] +nod_pos_tri[e,2,0])/3.0
            meanY = (nod_pos_tri[e,0,1] +nod_pos_tri[e,1,1] +nod_pos_tri[e,2,1])/3.0
            meanU = (unknown_vec_u[nod_num_tri[e,0]] +unknown_vec_u[nod_num_tri[e,1]] +unknown_vec_u[nod_num_tri[e,2]])/3.0
            ax.text(meanX, meanY, meanU, 'e%d' %e, ha='center', color='#000000')

    #グラフを表示
    if(out_type=='show'):
        plt.show()
    elif(out_type=='save'):
        plt.savefig("fem2d_poisson.png")
    plt.close()  #作成した図のウィンドウを消す


#メイン実行部
if __name__ == '__main__':
    ##### プリプロセス #####
    x_min = -1.0  #計算領域のXの最小値
    x_max = 1.0  #計算領域のXの最大値
    y_min = -1.0  #計算領域のYの最小値
    y_max = 1.0  #計算領域のYの最大値
    func_f = 1.0  #定数関数f

    #左部(x=x_min)、右部(x=x_max)、下部(y=y_min)、上部(y=y_max)の、境界の種類と値
    #境界の種類はDirichlet,Neumann
    BC_left = ['Dirichlet', 0.0]
    BC_right = ['Neumann', 1.0]
    BC_bottom = ['Neumann', 0.0]
    BC_top = ['Neumann', 0.0]

    #節点の生成方法。lattice,random
    node_type = ['lattice', 10]  #数字は格子分割におけるx・y方向の節点数
    #node_type = ['random', 50]  #数字はランダム分割における節点数
    matrix_type = 'sparse'  #全体行列の形式。basic,sparse

    #節点データ生成。Global節点座標、三角形要素の節点番号、境界線分要素の節点番号
    nod_pos_glo, nod_num_tri, nod_num_seg = generate_nodes(node_type)

    #Local節点座標を作成。三角形要素のLocal節点座標、境界線分要素のLocal節点座標
    nod_pos_tri, nod_pos_seg = make_mesh_data()

    #メッシュを表示(ポストプロセス)。番号などの有無(True,False)、グラフの表示方法(show,save)
    visualize_mesh(nod_pos_glo, show_text=True, out_type='show')


    ##### メインプロセス #####
    #計算の開始時刻を記録
    print ("Calculation start: ", time.ctime())  #計算開始時刻を表示
    compute_time = time.time()  #計算の開始時刻

    #要素行列の構築
    mat_A_ele, vec_b_ele, area_tri = assemble_element_matrix(nod_num_tri, nod_pos_tri)

    #全体行列の構築
    mat_A_glo, vec_b_glo = assemble_global_matrix(matrix_type)

    #境界要素の情報を設定
    BC_type, BC_value, leng_seg = make_boundary_info(nod_pos_seg)

    #境界条件を実装
    mat_A_glo, vec_b_glo = set_boundary_condition(mat_A_glo, vec_b_glo, BC_type, BC_value, leng_seg)

    #連立方程式を解く
    unknown_vec_u = solve_simultaneous_equations(mat_A_glo, vec_b_glo)

    #計算時間の表示
    compute_time = time.time() -compute_time
    print ("Calculation time: {:0.5f}[sec]".format(compute_time))

    #計算結果を表示(ポストプロセス)。番号などの有無(True,False)、グラフの表示方法(show,save)
    visualize_result(nod_pos_glo, unknown_vec_u, show_text=False, out_type='show')
