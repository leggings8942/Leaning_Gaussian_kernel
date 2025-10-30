from abc import ABCMeta, abstractmethod
from typing import List, Tuple
import numpy as np
import scipy as sp
from scipy.linalg import solve_triangular
from scipy.spatial import distance as spd
import pandas as pd
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor



# 修正コレスキー分解
def modified_cholesky(x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:        
    if x.ndim != 2:
        print(f"x dims = {x.ndim}")
        raise ValueError("エラー：：次元数が一致しません。")
    
    if x.shape[0] != x.shape[1]:
        print(f"x shape = {x.shape}")
        raise ValueError("エラー：：正方行列ではありません。")
    
    n = x.shape[0]
    d = np.diag(x).copy()
    L = np.tril(x, k=-1).copy() + np.identity(n)
    
    for idx1 in range(1, n):
        prev = idx1 - 1
        tmp  = d[0:prev] if d[0:prev].size != 0 else 0
        tmp  = np.dot(L[idx1:, 0:prev], (L[prev, 0:prev] * tmp).T)
        
        DIV  = d[prev] if d[prev] != 0 else 1e-16
        L[idx1:, prev] = (L[idx1:, prev] - tmp) / DIV
        d[idx1]       -= np.sum((L[idx1, 0:idx1] ** 2) * d[0:idx1])
    
    d = np.diag(d)
    return L, d

# 軟判別閾値関数
def soft_threshold(x:np.ndarray, α:float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - α, 0)

# 相関係数カーネルのインターフェース
class Kernel(metaclass=ABCMeta):
    @abstractmethod
    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError()

# 定数カーネル
class ConstantKernel(Kernel):
    def __init__(self, alpha:float):
        super().__init__()
        self.alpha = alpha
        return None

    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        if (type(DATA_X) is not np.ndarray) or (type(DATA_Y) not in {np.ndarray, type(None)}):
            print(f"type(DATA_X) = {type(DATA_X)}")
            print(f"type(DATA_Y) = {type(DATA_Y)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (DATA_X.ndim != 2) or ((type(DATA_Y) != type(None)) and (DATA_Y.ndim != 2)):
            print(f"DATA_X.ndim = {DATA_X.ndim}")
            print(f"DATA_Y.ndim = {DATA_Y.ndim}")
            raise ValueError("エラー：：次元数が一致しません。")
        
        # 対称行列であるか判定
        if np.allclose(DATA_X, DATA_X.T, atol=1e-8) and ((type(DATA_Y) != type(None)) and np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)):
            print(f"all(DATA_X == DATA_X.T) = {np.allclose(DATA_X, DATA_X.T, atol=1e-8)}")
            print(f"all(DATA_Y == DATA_Y.T) = {np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)}")
            raise ValueError("エラー：：対称行列ではありません。")
        
        if type(DATA_Y) == type(None):
            return np.full_like(DATA_X, self.alpha)
        else:
            return np.full_like(np.empty((DATA_X.shape[0], DATA_Y.shape[0])), self.alpha)

    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# 加算カーネル
class SumKernel(Kernel):
    def __init__(self, left:Kernel, right:Kernel) -> None:
        super().__init__()
        self.left  = left
        self.right = right
        return None
    
    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        if (type(DATA_X) is not np.ndarray) or (type(DATA_Y) not in {np.ndarray, type(None)}):
            print(f"type(DATA_X) = {type(DATA_X)}")
            print(f"type(DATA_Y) = {type(DATA_Y)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (DATA_X.ndim != 2) or ((type(DATA_Y) != type(None)) and (DATA_Y.ndim != 2)):
            print(f"DATA_X.ndim = {DATA_X.ndim}")
            print(f"DATA_Y.ndim = {DATA_Y.ndim}")
            raise ValueError("エラー：：次元数が一致しません。")
        
        # 対称行列であるか判定
        if np.allclose(DATA_X, DATA_X.T, atol=1e-8) and ((type(DATA_Y) != type(None)) and np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)):
            print(f"all(DATA_X == DATA_X.T) = {np.allclose(DATA_X, DATA_X.T, atol=1e-8)}")
            print(f"all(DATA_Y == DATA_Y.T) = {np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)}")
            raise ValueError("エラー：：対称行列ではありません。")

        return self.left.correlation(DATA_X, DATA_Y) + self.right.correlation(DATA_X, DATA_Y)

    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# ホワイトノイズカーネル
class WhiteNoiseKernel(Kernel):
    def __init__(self, alpha:float):
        super().__init__()
        self.alpha = alpha
        return None
    
    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        if (type(DATA_X) is not np.ndarray) or (type(DATA_Y) not in {np.ndarray, type(None)}):
            print(f"type(DATA_X) = {type(DATA_X)}")
            print(f"type(DATA_Y) = {type(DATA_Y)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (DATA_X.ndim != 2) or ((type(DATA_Y) != type(None)) and (DATA_Y.ndim != 2)):
            print(f"DATA_X.ndim = {DATA_X.ndim}")
            print(f"DATA_Y.ndim = {DATA_Y.ndim}")
            raise ValueError("エラー：：次元数が一致しません。")
        
        # 対称行列であるか判定
        if np.allclose(DATA_X, DATA_X.T, atol=1e-8) and ((type(DATA_Y) != type(None)) and np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)):
            print(f"all(DATA_X == DATA_X.T) = {np.allclose(DATA_X, DATA_X.T, atol=1e-8)}")
            print(f"all(DATA_Y == DATA_Y.T) = {np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)}")
            raise ValueError("エラー：：対称行列ではありません。")
    
        # クロネッカーのデルタを適用する
        if type(DATA_Y) == type(None):
            return self.alpha * np.eye(DATA_X.shape[0])
        else:
            return np.zeros((DATA_X.shape[0], DATA_Y.shape[0]))
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# 線形カーネル
class LinearKernel(Kernel):
    def __init__(self, alpha:float):
        super().__init__()
        self.alpha = alpha
        return None

    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        if (type(DATA_X) is not np.ndarray) or (type(DATA_Y) not in {np.ndarray, type(None)}):
            print(f"type(DATA_X) = {type(DATA_X)}")
            print(f"type(DATA_Y) = {type(DATA_Y)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (DATA_X.ndim != 2) or ((type(DATA_Y) != type(None)) and (DATA_Y.ndim != 2)):
            print(f"DATA_X.ndim = {DATA_X.ndim}")
            print(f"DATA_Y.ndim = {DATA_Y.ndim}")
            raise ValueError("エラー：：次元数が一致しません。")
        
        # 対称行列であるか判定
        if np.allclose(DATA_X, DATA_X.T, atol=1e-8) and ((type(DATA_Y) != type(None)) and np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)):
            print(f"all(DATA_X == DATA_X.T) = {np.allclose(DATA_X, DATA_X.T, atol=1e-8)}")
            print(f"all(DATA_Y == DATA_Y.T) = {np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)}")
            raise ValueError("エラー：：対称行列ではありません。")
    
        # 内積を計算する
        if type(DATA_Y) == type(None):
            K = DATA_X @ DATA_X.T
        else:
            K = DATA_X @ DATA_Y.T
        return self.alpha * K
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# 指数カーネル
class ExponentialKernel(Kernel):
    def __init__(self, alpha:float, beta:float):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        return None

    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        if (type(DATA_X) is not np.ndarray) or (type(DATA_Y) not in {np.ndarray, type(None)}):
            print(f"type(DATA_X) = {type(DATA_X)}")
            print(f"type(DATA_Y) = {type(DATA_Y)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (DATA_X.ndim != 2) or ((type(DATA_Y) != type(None)) and (DATA_Y.ndim != 2)):
            print(f"DATA_X.ndim = {DATA_X.ndim}")
            print(f"DATA_Y.ndim = {DATA_Y.ndim}")
            raise ValueError("エラー：：次元数が一致しません。")
        
        # 対称行列であるか判定
        if np.allclose(DATA_X, DATA_X.T, atol=1e-8) and ((type(DATA_Y) != type(None)) and np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)):
            print(f"all(DATA_X == DATA_X.T) = {np.allclose(DATA_X, DATA_X.T, atol=1e-8)}")
            print(f"all(DATA_Y == DATA_Y.T) = {np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)}")
            raise ValueError("エラー：：対称行列ではありません。")

        if type(DATA_Y) == type(None):
            K = spd.squareform(spd.pdist(DATA_X, metric="cityblock"))
            K = np.exp(K / self.alpha)
        else:
            K = spd.cdist(DATA_X, DATA_Y, metric="cityblock")
            K = np.exp(K / self.alpha)
        return self.beta * K
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# 周期カーネル
class PeriodicKernel(Kernel):
    def __init__(self, alpha:float, beta:float, gamma:float):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        return None

    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        if (type(DATA_X) is not np.ndarray) or (type(DATA_Y) not in {np.ndarray, type(None)}):
            print(f"type(DATA_X) = {type(DATA_X)}")
            print(f"type(DATA_Y) = {type(DATA_Y)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (DATA_X.ndim != 2) or ((type(DATA_Y) != type(None)) and (DATA_Y.ndim != 2)):
            print(f"DATA_X.ndim = {DATA_X.ndim}")
            print(f"DATA_Y.ndim = {DATA_Y.ndim}")
            raise ValueError("エラー：：次元数が一致しません。")
        
        # 対称行列であるか判定
        if np.allclose(DATA_X, DATA_X.T, atol=1e-8) and ((type(DATA_Y) != type(None)) and np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)):
            print(f"all(DATA_X == DATA_X.T) = {np.allclose(DATA_X, DATA_X.T, atol=1e-8)}")
            print(f"all(DATA_Y == DATA_Y.T) = {np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)}")
            raise ValueError("エラー：：対称行列ではありません。")

        if type(DATA_Y) == type(None):
            K = spd.squareform(spd.pdist(DATA_X, metric="cityblock"))
            K = np.exp(self.beta * np.cos(K / self.alpha))
        else:
            K = spd.cdist(DATA_X, DATA_Y, metric="cityblock")
            K = np.exp(self.beta * np.cos(K / self.alpha))
        return self.gamma * K
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# ガウスカーネル(RBFカーネル)
class GaussKernel(Kernel):
    def __init__(self, alpha:float, beta:float):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        return None

    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        if (type(DATA_X) is not np.ndarray) or (type(DATA_Y) not in {np.ndarray, type(None)}):
            print(f"type(DATA_X) = {type(DATA_X)}")
            print(f"type(DATA_Y) = {type(DATA_Y)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (DATA_X.ndim != 2) or ((type(DATA_Y) != type(None)) and (DATA_Y.ndim != 2)):
            print(f"DATA_X.ndim = {DATA_X.ndim}")
            print(f"DATA_Y.ndim = {DATA_Y.ndim}")
            raise ValueError("エラー：：次元数が一致しません。")
        
        # 対称行列であるか判定
        if np.allclose(DATA_X, DATA_X.T, atol=1e-8) and ((type(DATA_Y) != type(None)) and np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)):
            print(f"all(DATA_X == DATA_X.T) = {np.allclose(DATA_X, DATA_X.T, atol=1e-8)}")
            print(f"all(DATA_Y == DATA_Y.T) = {np.allclose(DATA_Y, DATA_Y.T, atol=1e-8)}")
            raise ValueError("エラー：：対称行列ではありません。")

        if type(DATA_Y) == type(None):
            K = spd.squareform(spd.pdist(DATA_X, metric="sqeuclidean"))
            K = np.exp(-K / self.alpha)
        else:
            K = spd.cdist(DATA_X, DATA_Y, metric="sqeuclidean")
            K = np.exp(-K / self.alpha)
        return self.beta * K
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)



class Gaussian_Process_Regression:
    def __init__(self,
                 vec_data_y:np.ndarray,              # 学習対象出力データY
                 mat_data_x:np.ndarray,              # 学習対象入力データX
                 kernel:Kernel,                      # 相関カーネル
                 norm_α:float=1.0,                   # L1・L2正則化パラメータの強さ
                 l1_ratio:float=0.1,                 # L1・L2正則化の強さ配分・比率
                 tol:float=1e-6,                     # 許容誤差
                 isStandardization:bool=True,        # 標準化処理の適用有無
                 max_iterate:int=300000,             # 最大ループ回数
                 random_state=None) -> None:         # 乱数のシード値
        if (type(vec_data_y) is not np.ndarray) or (type(mat_data_x) is not np.ndarray):
            print(f"type(vec_data_y) = {type(vec_data_y)}")
            print(f"type(mat_data_x) = {type(mat_data_x)}")
            print("エラー：：Numpy型である必要があります。")
            raise

        if (vec_data_y.ndim != 2) or (vec_data_y.shape[1] != 1) or (mat_data_x.ndim != 2):
            print(f"vec_data_y dims  = {vec_data_y.ndim}")
            print(f"vec_data_y shape = {vec_data_y.shape}")
            print(f"mat_data_x dims  = {mat_data_x.ndim}")
            print(f"mat_data_x shape = {mat_data_x.shape}")
            print("エラー：：次元数が一致しません。")
            raise

        self.vec_data_y        = np.copy(vec_data_y)
        self.mat_data_x        = np.copy(mat_data_x)
        self.kernel            = kernel
        self.tol               = tol
        self.norm_α            = np.abs(norm_α)
        self.l1_ratio          = np.where(l1_ratio < 0, 0, np.where(l1_ratio > 1, 1, l1_ratio))
        self.isStandardization = isStandardization
        self.max_iterate       = round(max_iterate)

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
        # ・sklearnライブラリに実装されているGaussianProcessRegressor(外部ライブラリ)
        # ・座標降下法アルゴリズム(CD: Coordinate Descent Algorithm)
        # ・メジャライザー最適化( ISTA: Iterative Shrinkage soft-Thresholding Algorithm)
        # ・メジャライザー最適化(FISTA: Fast Iterative Shrinkage soft-Thresholding Algorithm)
        # これらのアルゴリズムは全て同じ目的関数を最適化している
        # しかし、実際に同一のパラメータでパラメータ探索をさせても同一の解は得られない
        # これは、実装の細かな違いによるものであったり、解析解ではなく近似解が得られるためであったりする
        # 特にISTAは勾配降下法と同等の性質を有しているため、異なる近似解が得られる
        # すなわち実行のたびに異なる解が導かれるかつ極所最適解に落ち着くことがある
        # また、外部ライブラリとしてsklearn.gaussian_process.GaussianProcessRegressorを利用することもできる
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
            # GaussianProcessRegressorの外部ライブラリである
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)を行なっている
            # このオプションではsklearnに実装されているモデルに処理を投げることを行なっている
            # 注意点として、データの標準化処理にはデフォルトで対応している点が挙げられる
            # 逆に標準化処理をOFFにする方法がわからなかったため、デフォルトでONとする事とした
            self.model = GaussianProcessRegressor(
                                kernel=(kernels.ConstantKernel()   * kernels.DotProduct()
                                        + kernels.ConstantKernel() * kernels.Matern(nu=0.5)
                                        + kernels.ConstantKernel() * kernels.ExpSineSquared()
                                        + kernels.ConstantKernel() * kernels.RBF()
                                        + kernels.ConstantKernel() * kernels.WhiteKernel()), 
                                alpha=0, random_state=0)
            self.model.fit(x_data, y_data)
            self.alpha = self.model.alpha_
            
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
            K    = self.kernel.correlation(x_data)
            L, d = modified_cholesky(K)
            
            # L2NORM = l2_norm * np.identity(expvars + 1)
            # L2NORM[0:expvars, 0:expvars] = L2NORM[0:expvars, 0:expvars] / np.square(self.x_std_dev.reshape([1, expvars]))
            # L2NORM[expvars,   expvars]   = 0
            # L = np.dot(A.T, A) + L2NORM
            # R = np.dot(A.T, b)
            # D = np.diag(np.diag(L))
            # G = np.diag(L)
            # C = L - D
            z = solve_triangular(L,       y_data, lower=True)
            x = solve_triangular(d @ L.T, z,      lower=False)
            self.trian = L
            self.diag  = d
            self.alpha = x
            
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
            pass     
        
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
            pass
            
        else:
            raise

        self.learn_flg = True
        return self.learn_flg
    
    def predict(self, test_X:np.ndarray, return_std:bool=False, return_cov:bool=False) -> np.ndarray:
        x_data = self.mat_data_x
        x_test = test_X
        if self.isStandardization:
            # x軸の標準化
            self.x_mean    = np.mean(x_data, axis=0)
            self.x_std_dev = np.std( x_data, axis=0)
            self.x_std_dev[self.x_std_dev < 1e-32] = 1
            x_data = (x_data - self.x_mean) / self.x_std_dev
            x_test = (x_test - self.x_mean) / self.x_std_dev

        # 予測平均・予測分散の算出
        K_     = self.kernel.correlation(x_data, x_test)
        K__    = self.kernel.correlation(x_test)
        V      = self.trian.T @ K_
        y_pred = K_.T @ self.alpha
        y_var  = K__ - V.T @ self.diag @ V

        if self.isStandardization:
            # y軸の標準化をもとに戻す
            y_pred = self.y_std_dev * y_pred + self.y_mean
            y_var  = y_var * self.y_std_dev**2

        if return_cov:
            return y_pred, y_var
        elif return_std:
            y_std = np.sqrt(y_var)
            y_std = np.diag(y_std)
            return y_pred, y_std
        else:
            return y_pred




