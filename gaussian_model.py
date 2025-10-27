from typing import List
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor



# 軟判別閾値関数
def soft_threshold(x:np.ndarray, α:float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - α, 0)

# クロネッカーのデルタ
def kronecker_delta(N:int) -> np.ndarray:
	return np.eye(N)

# 線形カーネル
def linear_kernel(xi:np.ndarray, xj:np.ndarray) -> np.ndarray:
    return np.sum(xi * xj, axis=2)

# 指数カーネル
def exponential_kernel(xi:np.ndarray, xj:np.ndarray, beta:float=1) -> np.ndarray:
    return np.exp(-np.linalg.norm(xi - xj, ord=1, axis=2) / beta)

# 周期カーネル
def periodic_kernel(xi:np.ndarray, xj:np.ndarray, beta:List[float]=[1, 1]) -> np.ndarray:
	return np.exp(beta[0] * np.cos(np.linalg.norm(xi - xj, ord=1, axis=2) / beta[1]))

# ガウシアンカーネル
def gauss_kernel(xi:np.ndarray, xj:np.ndarray, beta:float=1) -> np.ndarray:
    return np.exp(-np.linalg.norm(xi - xj, ord=2, axis=2) / beta)

class Gaussian_Process_Regression:
    def __init__(self,
                 vec_data_y:np.ndarray,              # 学習対象出力データY
                 mat_data_x:np.ndarray,              # 学習対象入力データX
                 norm_α:float=1.0,                   # L1・L2正則化パラメータの強さ
                 l1_ratio:float=0.1,                 # L1・L2正則化の強さ配分・比率
                 tol:float=1e-6,                     # 許容誤差
                 isStandardization:bool=True,        # 標準化処理の適用有無
                 max_iterate:int=300000,             # 最大ループ回数
                 random_state=None) -> None:         # 乱数のシード値
        if type(vec_data_y) is not np.ndarray:
            print(f"type(vec_data_y) = {type(vec_data_y)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (vec_data_y.ndim != 2) or (vec_data_y.shape[1] == 1):
            print(f"vec_data_y dims  = {vec_data_y.ndim}")
            print(f"vec_data_y shape = {vec_data_y.shape}")
            print("エラー：：次元数が一致しません。")
            raise

        if type(mat_data_x) is not np.ndarray:
            print(f"type(mat_data_x) = {type(mat_data_x)}")
            print("エラー：：Numpy型である必要があります。")
            raise
        
        if mat_data_x.ndim != 2:
            print(f"mat_data_x dims  = {mat_data_x.ndim}")
            print(f"mat_data_x shape = {mat_data_x.shape}")
            print("エラー：：次元数が一致しません。")
            raise


        self.vec_data_y          = vec_data_y
        self.mat_data_x          = mat_data_x
        self.tol                 = tol
        self.norm_α              = np.abs(norm_α)
        self.l1_ratio            = np.where(l1_ratio < 0, 0, np.where(l1_ratio > 1, 1, l1_ratio))
        self.isStandardization   = isStandardization
        self.max_iterate         = round(max_iterate)

        self.random_state = random_state
        if random_state != None:
            self.random = np.random
            self.random.seed(seed=self.random_state)
        else:
            self.random = np.random


    def fit(self, solver:str='external library', visible_flg:bool=False) -> bool:
        # 本ライブラリにおいてエラーの出力を行わないのは、近似的にでも処理結果が欲しいためである
        # また、solverとしてISTA・FISTAを使用する際にも注意が必要である
        # ISTA・FISTAは勾配降下法に似た特徴を有しており、対象の最適化パラメータのスケールに弱い
        # 最適化対象のパラメータの解析解のスケールに依存して、必要な更新回数が多くなる
        # スケールが極端に大きい場合などには事実上収束しないが、そもそも解析解のスケールを事前に知らない・気にしていない場合も多い
        # そのような場合には、教師データ(X, Y)をそれぞれ標準化することで対処できる
        # isStandardization=True に設定しておくことを強く推奨する
        
        x_data = self.mat_data_x
        y_data = self.vec_data_y
        
        data_num, expvars = x_data.shape
        _,        objvars = y_data.shape
        
        # 標準化指定の有無
        if self.isStandardization:
            # x軸の標準化
            self.x_mean    = np.mean(x_data, axis=0)
            self.x_std_dev = np.std( x_data, axis=0)
            self.x_std_dev[self.x_std_dev < 1e-32] = 1
            x_data = (x_data - self.x_mean) / self.x_std_dev
            
            # y軸の標準化
            self.y_mean    = np.mean(y_data, axis=0)
            self.y_std_dev = np.std( y_data, axis=0)
            self.y_std_dev[self.y_std_dev < 1e-32] = 1
            y_data = (y_data - self.y_mean) / self.y_std_dev
        else:
            self.x_mean    = np.zeros(expvars)
            self.x_std_dev = np.ones( expvars)
            self.y_mean    = np.zeros(objvars)
            self.y_std_dev = np.ones( objvars)
            
        
        # 本ライブラリで実装されているアルゴリズムは以下の4点となる
        # ・sklearnライブラリに実装されているElasticNet(外部ライブラリ)
        # ・座標降下法アルゴリズム(CD: Coordinate Descent Algorithm)
        # ・メジャライザー最適化( ISTA: Iterative Shrinkage soft-Thresholding Algorithm)
        # ・メジャライザー最適化(FISTA: Fast Iterative Shrinkage soft-Thresholding Algorithm)
        # これらのアルゴリズムは全て同じ目的関数を最適化している
        # しかし、実際に同一のパラメータでパラメータ探索をさせても同一の解は得られない
        # これは、実装の細かな違いによるものであったり、解析解ではなく近似解が得られるためであったりする
        # 特にISTAは勾配降下法と同等の性質を有しているため、異なる近似解が得られる
        # すなわち実行のたびに異なる解が導かれるかつ極所最適解に落ち着くことがある
        # また、外部ライブラリとしてsklearn.linear_model.ElasticNetを利用することもできる
        # この外部ライブラリは内部で座標降下法で探索を行っている点で本ライブラリと同等である
        # 一方で、この外部ライブラリはC言語(Cython)を利用してチューニングが行われている
        # また広く公開され、多くの人に利用されているライブラリでもあるため速度・品質ともにレベルが高い
        # 探索解の品質を保証したいのであれば、外部ライブラリの利用を強く推奨する
        # 一方で、外部ライブラリはデータの標準化処理に対応していない点に注意する必要がある
        # データの標準化処理を行う場合にはL1・L2正則化項の調整を行う必要があるが、外部ライブラリでは行うことができないためである
        # 最後に広く認められているわけではないため使用の際には注意が必要であるが、本ライブラリにて実装済みの
        # これら3種類のアルゴリズムが想定する目的関数は以下のとおり
        # A = 説明変数x + 切片b の行列(データ数n ✖️ (説明変数数s + 1))
        # B = 目的変数y の行列(データ数n ✖️ 目的変数数m)
        # X = 説明変数xの係数 + 切片bの係数 の行列((説明変数数s + 1) ✖️ 目的変数数m)
        # λ_1 = 正則化の強度 * l1_ratio
        # λ_2 = 正則化の強度 * (1 - l1_ratio)
        # math: \begin{equation}
        # math: \begin{split}
        # math: Objective &= \frac{1}{n} \| B - AX \|_2^2 + \frac{λ_2}{2} \| X \|_2^2 + λ_1 \|X\|_1 \\
        # math: &= tr [ \left( B - AX \right) ^T \left( B - AX \right) ] + \frac{λ_2 n}{2} tr [ X^T X ] + λ_1 n \sum_{i=1} |x_i |
        # math: \end{split}
        # math: \end{equation}
        # 参考までに各オプションごとの実行速度は以下の通り
        # external library  >>  FISTA  >>  ISTA  >>  coordinate descent
        
        if   solver == "external library":
            # ElasticNetの外部ライブラリである
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)を行なっている
            # このオプションではsklearnに実装されているモデルに処理を投げることを行なっている
            # 注意点として、データの標準化処理には対応していないことが挙げられる
            # 仮に標準化処理付きでこのオプションが選択された場合には、簡易的な正則化項の調整を行うことにしている
            # しかし、この調整は非常に簡素なものであり厳密性に欠ける
            # このオプションを利用する際には、標準化処理を行わないことを強く推奨する
            self.model = GaussianProcessRegressor(
                                kernel=(kernels.ConstantKernel()   * kernels.DotProduct()
                                        + kernels.ConstantKernel() * kernels.Matern(nu=0.5)
                                        + kernels.ConstantKernel() * kernels.ExpSineSquared()
                                        + kernels.ConstantKernel() * kernels.RBF()
                                        + kernels.ConstantKernel() * kernels.WhiteKernel()), 
                                alpha=0, random_state=0)
            self.model.fit(x_data, y_data)
            
            if visible_flg:
                l1_norm = self.norm_α * self.l1_ratio       * data_num
                l2_norm = self.norm_α * (1 - self.l1_ratio) * data_num
                A       = np.hstack([x_data, np.ones([data_num, 1])])
                B       = y_data
                X       = np.vstack([self.alpha, self.alpha0])
                DIFF = B - np.dot(A, X)
                DIFF = np.dot(DIFF.T, DIFF)
                SQUA = np.dot(X.T, X)
                SQUA[objvars-1, objvars-1] = 0
                ABSO = np.abs(X)
                ABSO[expvars, :] = 0
                OBJE = 1 / 2 * np.sum(np.diag(DIFF)) + l2_norm / 2 * np.sum(np.diag(SQUA)) + l1_norm * np.sum(ABSO)
                print("平均二乗誤差(MSE):", np.sum(np.diag(DIFF)) / data_num, flush=True)
                print("L2正則化項(l2 norm):", np.sum(np.diag(SQUA)))
                print("L1正則化項(l1 norm):", np.sum(ABSO))
                print("目的関数(Objective): ", OBJE)
                
                X     = np.vstack([self.alpha, np.zeros([1, objvars])])
                DLoss = np.dot(A.T, B) - l1_norm * np.sign(X) - np.dot(np.dot(A.T, A) + l2_norm * np.identity(expvars + 1), X)
                print("目的関数(Objective)の微分: ", np.abs(DLoss).sum())
            
        elif solver == "coordinate descent":
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)を行なっている
            # 注意点として、切片に対してはラッソ最適化を行わないことが挙げられる
            # リッジ最適化は一般に係数を0にするためではなく、最適化対象のパラメータ全体を小さく保つために利用される
            # 一方で、ラッソ最適化は係数を0にするために利用される手法である
            # そのため、一般にはラッソ最適化を切片に対しては適用しない習慣がある
            # このライブラリもこの習慣に従うことにする
            # リッジ最適化についても切片に対しては適用しないことにした
            # これは標準化を行う前と後で、正則化の効果が変動してしまうことを防ぐためである
            # 実装アルゴリズムは座標降下法である
            # できる限り高速に処理を行いたかったので、このような実装になった
            # このアルゴリズムの計算量は、O(ループ回数 × 説明変数の数 × O(行列積))である
            # 1×M, M×Lの大きさを持つ行列A, Bを想定すると、行列積の計算量はO(ML)となる
            # このSVARライブラリではそれぞれ、M=(説明変数の数 + 1) L=目的変数の数に対応している
            # 計算量オーダーを書き直すと O(ループ回数 × 説明変数の数 × ML)となる
            # このアルゴリズムを利用するにあたって、学習対象データの標準化などの条件は特にない
            # しかし多くの場合において、標準化処理を施してある学習データに対する学習速度は早い
            # その意味で標準化処理を推奨する
            l1_norm = self.norm_α * self.l1_ratio       * data_num
            l2_norm = self.norm_α * (1 - self.l1_ratio) * data_num
            A       = np.hstack([x_data, np.ones([data_num, 1])])
            b       = y_data
            
            L2NORM = l2_norm * np.identity(expvars + 1)
            L2NORM[0:expvars, 0:expvars] = L2NORM[0:expvars, 0:expvars] / np.square(self.x_std_dev.reshape([1, expvars]))
            L2NORM[expvars,   expvars]   = 0
            L = np.dot(A.T, A) + L2NORM
            R = np.dot(A.T, b)
            D = np.diag(np.diag(L))
            G = np.diag(L)
            C = L - D
            
            x_new = np.zeros([expvars + 1, objvars])
            for idx1 in range(0, self.max_iterate):
                x_old = x_new.copy()
                
                x_new[expvars, :] = (R[expvars, :] - np.dot(C[expvars, :], x_new)) / G[expvars]
                for idx2 in range(0, expvars):
                    tmp = R[idx2, :] - np.dot(C[idx2, :], x_new)
                    x_new[idx2, :] = soft_threshold(tmp, l1_norm / self.x_std_dev[idx2] / self.y_std_dev) / G[idx2]
                
                ΔDiff = np.sum((x_new - x_old) ** 2) / data_num
                if visible_flg and (idx1 % 1000 == 0):
                    print(f"ite:{idx1+1}  ΔDiff:{ΔDiff}")
                
                if ΔDiff <= self.tol:
                    break
            
            x = x_new
            self.alpha, self.alpha0 = x[0:expvars, :], x[expvars, :]
            self.alpha0 = self.alpha0.reshape([1, x.shape[1]])
            
            if visible_flg:
                l1_norm = self.norm_α * self.l1_ratio       * data_num
                l2_norm = self.norm_α * (1 - self.l1_ratio) * data_num
                A       = np.hstack([x_data, np.ones([data_num, 1])])
                B       = y_data
                X       = np.vstack([self.alpha, self.alpha0])
                DIFF = B - np.dot(A, X)
                DIFF = np.dot(DIFF.T, DIFF)
                SQUA = np.dot(X.T, X)
                SQUA[objvars-1, objvars-1] = 0
                ABSO = np.abs(X)
                ABSO[expvars, :] = 0
                OBJE = 1 / 2 * np.sum(np.diag(DIFF)) + l2_norm / 2 * np.sum(np.diag(SQUA)) + l1_norm * np.sum(ABSO)
                print("平均二乗誤差(MSE):", np.sum(np.diag(DIFF)) / data_num, flush=True)
                print("L2正則化項(l2 norm):", np.sum(np.diag(SQUA)))
                print("L1正則化項(l1 norm):", np.sum(ABSO))
                print("目的関数(Objective): ", OBJE)
                
                X     = np.vstack([self.alpha, np.zeros([1, objvars])])
                DLoss = np.dot(A.T, B) - l1_norm * np.sign(X) - np.dot(np.dot(A.T, A) + l2_norm * np.identity(expvars + 1), X)
                print("目的関数(Objective)の微分: ", np.abs(DLoss).sum())
            
        elif solver == "ISTA":
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)を行なっている
            # 注意点として、切片に対してはラッソ最適化を行わないことが挙げられる
            # リッジ最適化は一般に係数を0にするためではなく、最適化対象のパラメータ全体を小さく保つために利用される
            # 一方で、ラッソ最適化は係数を0にするために利用される手法である
            # そのため、一般にはラッソ最適化を切片に対しては適用しない習慣がある
            # このライブラリもこの習慣に従うことにする
            # リッジ最適化についても切片に対しては適用しないことにした
            # これは標準化を行う前と後で、正則化の効果が変動してしまうことを防ぐためである
            # 実装アルゴリズムは一般的なメジャライザー最適化(ISTA: Iterative Shrinkage soft-Thresholding Algorithm)である
            # このアルゴリズムのメジャライザー部分は勾配降下法の更新式に等しい
            # このアルゴリズムを利用する際の注意点として、以下の２つが挙げられる
            # ・教師データ(X, Y)がそれぞれ標準化されている必要があること
            # ・設定イレーション回数が十分でない場合に、大域的最適解への収束が保証できないこと
            # 標準化されていない場合にはうまく収束しないくなる等、アルゴリズムが機能しなくなる可能性がある
            # isStandardization=True に設定しておけば、問題ない
            
            l1_norm      = self.norm_α * self.l1_ratio       * data_num
            l2_norm      = self.norm_α * (1 - self.l1_ratio) * data_num
            L            = np.linalg.norm(A.T.dot(A), ord="fro")
            theta_new    = self.random.random()
            l1_specifier = np.ones(x_new.shape)
            l2_specifier = np.ones(x_new.shape)
            l1_specifier[0:expvars, :] = l1_specifier[0:expvars, :] / self.x_std_dev.reshape([expvars, 1])            / self.y_std_dev.reshape([1, objvars])
            l2_specifier[0:expvars, :] = l2_specifier[0:expvars, :] / np.square(self.x_std_dev.reshape([expvars, 1]))
            l1_specifier[expvars,   :] = 0
            l2_specifier[expvars,   :] = 0
            Base_Loss    = 0
            for idx in range(0, self.max_iterate):
                INPUT_D = np.tile(x_data, (data_num, 1))
                K_linea = beta[0] * linear_kernel(     INPUT_D, INPUT_D.T)
                K_expon = beta[1] * exponential_kernel(xi, xj, beta=beta[2])
                K_perio = beta[3] * periodic_kernel(   xi, xj, beta=[beta[4], beta[5]])
                K_gauss = beta[6] * gauss_kernel(      xi, xj, beta=beta[7])
                K_noise = beta[8] * kronecker_delta(   xi.shape[0])
                ΔLoss  = b - np.dot(A, x_new)
                ΔDiff  = np.dot(A.T, ΔLoss)
                
                rho    = 1 / L
                diff_x = rho * ΔDiff
                x_new  = soft_threshold(x_new + diff_x, rho * l1_norm * l1_specifier)
                x_new  = x_new / (1 + rho * l2_norm * l2_specifier)
                
                mse = np.sum(ΔLoss ** 2)
                if visible_flg and (idx % 1000 == 0):
                    update_diff = np.sum(diff_x ** 2)
                    print(f"ite:{idx+1}  mse:{mse}  update_diff:{update_diff} diff:{np.abs(Base_Loss - mse)}")
                
                if np.abs(Base_Loss - mse) <= self.tol:
                    break
                else:
                    Base_Loss = mse
            
            x = x_new
            self.alpha, self.alpha0 = x[0:expvars, :], x[expvars, :]
            self.alpha0 = self.alpha0.reshape([1, x.shape[1]])
            
            if visible_flg:
                l1_norm = self.norm_α * self.l1_ratio       * data_num
                l2_norm = self.norm_α * (1 - self.l1_ratio) * data_num
                A       = np.hstack([x_data, np.ones([data_num, 1])])
                B       = y_data
                X       = np.vstack([self.alpha, self.alpha0])
                DIFF = B - np.dot(A, X)
                DIFF = np.dot(DIFF.T, DIFF)
                SQUA = np.dot(X.T, X)
                SQUA[objvars-1, objvars-1] = 0
                ABSO = np.abs(X)
                ABSO[expvars, :] = 0
                OBJE = 1 / 2 * np.sum(np.diag(DIFF)) + l2_norm / 2 * np.sum(np.diag(SQUA)) + l1_norm * np.sum(ABSO)
                print("平均二乗誤差(MSE):", np.sum(np.diag(DIFF)) / data_num, flush=True)
                print("L2正則化項(l2 norm):", np.sum(np.diag(SQUA)))
                print("L1正則化項(l1 norm):", np.sum(ABSO))
                print("目的関数(Objective): ", OBJE)
                
                X     = np.vstack([self.alpha, np.zeros([1, objvars])])
                DLoss = np.dot(A.T, B) - l1_norm * np.sign(X) - np.dot(np.dot(A.T, A) + l2_norm * np.identity(expvars + 1), X)
                print("目的関数(Objective)の微分: ", np.abs(DLoss).sum())
        
        elif solver == "FISTA":
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)を行なっている
            # 注意点として、切片に対してはラッソ最適化を行わないことが挙げられる
            # リッジ最適化は一般に係数を0にするためではなく、最適化対象のパラメータ全体を小さく保つために利用される
            # 一方で、ラッソ最適化は係数を0にするために利用される手法である
            # そのため、一般にはラッソ最適化を切片に対しては適用しない習慣がある
            # このライブラリもこの習慣に従うことにする
            # リッジ最適化についても切片に対しては適用しないことにした
            # これは標準化を行う前と後で、正則化の効果が変動してしまうことを防ぐためである
            # 実装アルゴリズムは一般的なメジャライザー最適化(FISTA: Fast Iterative Shrinkage soft-Thresholding Algorithm)である
            # このアルゴリズムのメジャライザー部分は勾配降下法の更新式に等しい
            # このアルゴリズムを利用する際の注意点として、以下の２つが挙げられる
            # ・教師データ(X, Y)がそれぞれ標準化されている必要があること
            # ・設定イレーション回数が十分でない場合に、大域的最適解への収束が保証できないこと
            # 標準化されていない場合にはうまく収束しないくなる等、アルゴリズムが機能しなくなる可能性がある
            # isStandardization=True に設定しておけば、問題ない
            
            l1_norm      = self.norm_α * self.l1_ratio       * data_num
            l2_norm      = self.norm_α * (1 - self.l1_ratio) * data_num
            A            = np.hstack([x_data, np.ones([data_num, 1])])
            b            = y_data
            L            = np.linalg.norm(A.T.dot(A), ord="fro")
            x_new        = self.random.random([A.shape[1], b.shape[1]])
            l1_specifier = np.ones(x_new.shape)
            l2_specifier = np.ones(x_new.shape)
            l1_specifier[0:expvars, :] = l1_specifier[0:expvars, :] / self.x_std_dev.reshape([expvars, 1])            / self.y_std_dev.reshape([1, objvars])
            l2_specifier[0:expvars, :] = l2_specifier[0:expvars, :] / np.square(self.x_std_dev.reshape([expvars, 1]))
            l1_specifier[expvars,   :] = 0
            l2_specifier[expvars,   :] = 0
            x_k_m_1      = x_new.copy()
            time_k       = 0
            Base_Loss    = 0
            for idx in range(0, self.max_iterate):
                ΔLoss  = b - np.dot(A, x_new)
                ΔDiff  = np.dot(A.T, ΔLoss)
                
                rho    = 1 / L
                diff_x = rho * ΔDiff
                x_tmp  = soft_threshold(x_new + diff_x, rho * l1_norm * l1_specifier)
                x_tmp  = x_tmp / (1 + rho * l2_norm * l2_specifier)
                
                time_k_a_1 = (1 + np.sqrt(1 + 4 * (time_k ** 2))) / 2
                x_new      = x_tmp + (time_k - 1) / (time_k_a_1) * (x_tmp - x_k_m_1)
                
                time_k  = time_k_a_1
                x_k_m_1 = x_tmp
                
                mse = np.sum(ΔLoss ** 2)
                if visible_flg and (idx % 1000 == 0):
                    update_diff = np.sum(diff_x ** 2)
                    print(f"ite:{idx+1}  mse:{mse}  update_diff:{update_diff} diff:{np.abs(Base_Loss - mse)}")
                
                if (idx != 1) and (np.abs(Base_Loss - mse) <= self.tol):
                    x_new = x_k_m_1
                    break
                else:
                    Base_Loss = mse
            
            x = x_new
            self.alpha, self.alpha0 = x[0:expvars, :], x[expvars, :]
            self.alpha0 = self.alpha0.reshape([1, x.shape[1]])
            
            if visible_flg:
                l1_norm = self.norm_α * self.l1_ratio       * data_num
                l2_norm = self.norm_α * (1 - self.l1_ratio) * data_num
                A       = np.hstack([x_data, np.ones([data_num, 1])])
                B       = y_data
                X       = np.vstack([self.alpha, self.alpha0])
                DIFF = B - np.dot(A, X)
                DIFF = np.dot(DIFF.T, DIFF)
                SQUA = np.dot(X.T, X)
                SQUA[objvars-1, objvars-1] = 0
                ABSO = np.abs(X)
                ABSO[expvars, :] = 0
                OBJE = 1 / 2 * np.sum(np.diag(DIFF)) + l2_norm / 2 * np.sum(np.diag(SQUA)) + l1_norm * np.sum(ABSO)
                print("平均二乗誤差(MSE):", np.sum(np.diag(DIFF)) / data_num, flush=True)
                print("L2正則化項(l2 norm):", np.sum(np.diag(SQUA)))
                print("L1正則化項(l1 norm):", np.sum(ABSO))
                print("目的関数(Objective): ", OBJE)
                
                X     = np.vstack([self.alpha, np.zeros([1, objvars])])
                DLoss = np.dot(A.T, B) - l1_norm * np.sign(X) - np.dot(np.dot(A.T, A) + l2_norm * np.identity(expvars + 1), X)
                print("目的関数(Objective)の微分: ", np.abs(DLoss).sum())
            
        else:
            raise

        self.learn_flg = True
        return self.learn_flg
    
    def predict(self, test_X:np.ndarray) -> np.ndarray:
        pred_mu_ss, pred_sigma_ss = self.model.predict(test_X.reshape((-1, 1)), return_std=True)

        return pred_mu_ss, pred_sigma_ss




