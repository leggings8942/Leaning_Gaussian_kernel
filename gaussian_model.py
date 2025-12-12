from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from functools import reduce
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
    
    if not np.allclose(x, x.T, atol=1e-10):
        # 対称性をチェック
        raise ValueError("エラー：：対称行列ではありません。")
    
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
    
    # d = np.diag(d)
    return L, d

# 修正コレスキー分解による連立方程式のソルバー
def modcho_solve(A:np.ndarray, b:np.ndarray):
    L, d = modified_cholesky(A)
    z = solve_triangular(L,   b, lower=True,  unit_diagonal=True)
    w = z / d[:, np.newaxis]
    x = solve_triangular(L.T, w, lower=False, unit_diagonal=True)
    return x

# 軟判別閾値関数
def soft_threshold(x:np.ndarray, α:float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - α, 0)

# 対数尤度関数の微分量を返す関数
def diff_log_likelihood(K:np.ndarray, y:np.ndarray, dKθ:Tuple[np.ndarray]) -> np.ndarray:
    K_inv  = np.linalg.inv(K)
    solv_x = K_inv @ y
    ΔDiff  = tuple(-np.trace(K_inv @ diff_θ) + solv_x.T @ diff_θ @ solv_x for diff_θ in dKθ)
    ΔDiff  = tuple(diff_θ.item() for diff_θ in ΔDiff)
    return ΔDiff


class Update_RAdam:
    def __init__(self, alpha=0.0001, beta1=0.999, beta2=0.9999):
        self.alpha   = alpha
        self.beta1   = beta1
        self.beta2   = beta2
        self.time    = 0
        self.beta1_t = 1
        self.beta2_t = 1
        self.m       = 0.0
        self.v       = 0.0
        self.ρ_inf   = 0

    def update(self, grads):
        if self.time == 0:
            self.m      = 0.0
            self.v      = 0.0
            self.ρ_inf  = 2 / (1 - self.beta2) - 1
        
        ε = 1e-32
        self.time    += 1
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta1_t)
        ρ_t   = self.ρ_inf - 2 * self.time * self.beta2_t / (1 - self.beta2_t)

        if ρ_t > 4:
            v_hat = np.sqrt((1 - self.beta2_t) / self.v)
            r_t   = np.sqrt((ρ_t - 4) * (ρ_t - 2) * self.ρ_inf) / np.sqrt((self.ρ_inf - 4) * (self.ρ_inf - 2) * ρ_t)
            tmp_alpha = self.alpha * r_t * v_hat
        
        else:
            tmp_alpha = self.alpha

        return tmp_alpha, m_hat


def soft_maximum(x, α):
    if x >= 0:
        return  np.abs(x) + α
    else:
        return -np.abs(x) + α

class Update_Rafael:
    def __init__(self, alpha=0.0001, beta=0.999, isSHC=False):
        self.alpha  = alpha
        self.beta   = beta
        self.time   = 0
        self.beta_t = 1
        self.m      = 0.0
        self.v      = 0.0
        self.w      = 0.0
        self.σ_coef = 0
        self.isSHC = isSHC

    def update(self, grads):
        if self.time == 0:
            self.m      = 0.0
            self.v      = 0.0
            self.w      = 0.0
            self.σ_coef = (1 + self.beta) / 2
        
        ε = 1e-32
        self.time   += 1
        self.beta_t *= self.beta

        self.m = self.beta * self.m + (1 - self.beta) * grads
        m_hat = self.m / (1 - self.beta_t)

        self.v = self.beta * self.v + (1 - self.beta) * (grads ** 2)
        self.w = self.beta * self.w + (1 - self.beta) * ((grads / soft_maximum(m_hat, ε) - 1) ** 2)
        
        if self.beta - self.beta_t > 0.1:
            v_hat  = self.v * self.σ_coef / (self.beta - self.beta_t)
            w_hat  = self.w * self.σ_coef / (self.beta - self.beta_t)
            σ_com  = np.sqrt((v_hat + w_hat + ε) / 2)
            # σ_hes  = np.sqrt(w_hat + ε)
            
            # self-healing canonicalization
            R = 0
            if self.isSHC:
                def chebyshev(r):
                    tmp1 = σ_com + r
                    tmp2 = np.square(m_hat / tmp1)
                    f    =     np.sum(tmp2,                   axis=0) - r
                    df   = 2 * np.sum(tmp2 / tmp1,            axis=0) + 1
                    ddf  = 6 * np.sum(tmp2 / np.square(tmp1), axis=0)
                    newt = f / df
                    return r + newt + ddf / (2 * df) * np.square(newt)
                
                r_min = np.sum(np.square(m_hat / σ_com), axis=0)
                r_max = np.cbrt(np.sum(np.square(m_hat), axis=0))
                R = np.maximum(np.minimum(r_max, r_min), 1)
                R = chebyshev(R)
                # R = chebyshev(R)     # option: 精度を求めるならチェビシェフ法を2回適用する
                R = np.maximum(R, 1) # option: 収束速度は遅くなるが、安定性が向上する
            
            tmp_alpha = self.alpha / (σ_com + R)
            output    = m_hat
        else:
            tmp_alpha = self.alpha
            output    = np.sign(grads)
        
        return tmp_alpha, output


# 相関係数カーネルのインターフェース
class Kernel(metaclass=ABCMeta):
    @abstractmethod
    def correlation(self, DATA_X:np.ndarray, DATA_Y:np.ndarray=None) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def diff_theta(self) -> Tuple:
        raise NotImplementedError()
    
    @abstractmethod
    def update_theta(self, Δtheta:Tuple, eta:float=1e-6, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:
        raise NotImplementedError()
    
    @abstractmethod
    def get_theta(self) -> Tuple:
        raise NotImplementedError()
    
    @abstractmethod
    def set_theta(self, thetas:Tuple) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError()

# 定数カーネル
class ConstantKernel(Kernel):
    def __init__(self, alpha:float, isL1Reg:bool=True, isL2Reg:bool=True, isNonNeg:bool=True):
        super().__init__()
        self.alpha    = alpha
        self.isL1Reg  = isL1Reg
        self.isL2Reg  = isL2Reg
        self.isNonNeg = isNonNeg
        self.cache    = np.zeros([1, 1])
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
            self.cache = np.full_like(np.empty((DATA_X.shape[0], DATA_X.shape[0])), 1.0)
        else:
            self.cache = np.full_like(np.empty((DATA_X.shape[0], DATA_Y.shape[0])), 1.0)
        
        return self.alpha * self.cache

    def diff_theta(self) -> Tuple:
        diff_alpha = self.cache
        return (diff_alpha,)
    
    def update_theta(self, Δtheta:Tuple, eta:Tuple, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:
        # L1正則化が有効か
        if self.isL1Reg:
            Δalpha = soft_threshold(self.alpha + eta[0] * Δtheta[0], eta[0] * l1_norm)
        else:
            Δalpha = self.alpha + eta[0] * Δtheta[0]
        
        # L2正則化が有効か
        if self.isL2Reg:
            Δalpha = Δalpha / (1 + eta[0] * l2_norm)
        
        # 非負制約が有効か
        if self.isNonNeg:
            Δalpha = np.maximum(Δalpha, 0)
        
        self.alpha = Δalpha
        return True
    
    def get_theta(self) -> Tuple:
        return (self.alpha,)

    def set_theta(self, thetas:Tuple) -> bool:
        self.alpha = thetas[0]
        return True

    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# 加算カーネル
class SumKernel(Kernel):
    def __init__(self, left:Kernel, right:Kernel) -> None:
        super().__init__()

        self.kernels = ()
        self.kernels += left.kernels  if isinstance(left,  SumKernel) else (left,)
        self.kernels += right.kernels if isinstance(right, SumKernel) else (right,)
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

        return np.sum([elem.correlation(DATA_X, DATA_Y) for elem in self.kernels], axis=0)
    
    def diff_theta(self) -> Tuple:
        diff_tuple = reduce(lambda x, y: x + y, [elem.diff_theta() for elem in self.kernels])
        return diff_tuple
    
    def update_theta(self, Δtheta:Tuple, eta:Tuple, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:
        # それぞれの枝における葉ノードのハイパーパラメータの数が不明であるため
        # これの探索を行う
        tmp_tuple = Δtheta
        for elem in self.kernels:
            theta_num = len(elem.get_theta())
            elem.update_theta(tmp_tuple[0:theta_num], eta=eta[0:theta_num], l1_norm=l1_norm, l2_norm=l2_norm)
            tmp_tuple = tmp_tuple[theta_num:]
        return True
    
    def get_theta(self) -> Tuple:
        theta_tuple = reduce(lambda x, y: x + y, [elem.get_theta() for elem in self.kernels])
        return theta_tuple
    
    def set_theta(self, thetas:Tuple) -> bool:
        # それぞれの枝における葉ノードのハイパーパラメータの数が不明であるため
        # これの探索を行う
        tmp_tuple = thetas
        for elem in self.kernels:
            theta_num = len(elem.get_theta())
            elem.set_theta(tmp_tuple[0:theta_num])
            tmp_tuple = tmp_tuple[theta_num:]
        return True

    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# ホワイトノイズカーネル
class WhiteNoiseKernel(Kernel):
    def __init__(self, alpha:float):
        super().__init__()
        self.alpha = alpha
        self.cache = np.zeros([1, 1])
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

        # メモ：
        # ホワイトノイズカーネルのハイパーパラメータだけは0にも負にもなってはいけない
        # それは、誤差量が0以下である事を表現することに他ならないためである
        # 必ず、正の誤差量が存在するようにしなければならない
        # このことからy=1/2 x^2という変数変換を施す事とした
        tmp_alpha = (self.alpha ** 2) / 2

        # クロネッカーのデルタを適用する
        if type(DATA_Y) == type(None):
            self.cache = np.eye(DATA_X.shape[0])
        else:
            self.cache = np.zeros((DATA_X.shape[0], DATA_Y.shape[0]))
        
        return tmp_alpha * self.cache
    
    def diff_theta(self) -> Tuple:
        diff_alpha = self.cache * self.alpha
        return (diff_alpha,)
    
    def update_theta(self, Δtheta:Tuple, eta:Tuple, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:        
        self.alpha = self.alpha + eta[0] * Δtheta[0]
        return True
    
    def get_theta(self) -> Tuple:
        # メモ：
        # ホワイトノイズカーネルのハイパーパラメータだけは0にも負にもなってはいけない
        # それは、誤差量が0以下である事を表現することに他ならないためである
        # 必ず、正の誤差量が存在するようにしなければならない
        # このことからy=1/2 x^2という変数変換を施す事とした
        tmp_alpha = (self.alpha ** 2) / 2
        return (tmp_alpha,)
    
    def set_theta(self, thetas:Tuple) -> bool:
        if thetas[0] < 0:
            raise ValueError("The argument must be positive")
        self.alpha = np.sqrt(thetas[0] * 2)
        return True
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# ガウスカーネル(RBFカーネル)
class GaussKernel(Kernel):
    def __init__(self, alpha:float, beta:float, isL1Reg:bool=True, isL2Reg:bool=True, isNonNeg:bool=True):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.isL1Reg   = isL1Reg
        self.isL2Reg   = isL2Reg
        self.isNonNeg  = isNonNeg
        self.cache_mol = np.zeros([1, 1])
        self.cache_cor = np.zeros([1, 1])
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
            self.cache_mol = spd.squareform(spd.pdist(DATA_X, metric="sqeuclidean"))
            self.cache_cor = np.exp(-self.beta * self.cache_mol)
        else:
            self.cache_mol = spd.cdist(DATA_X, DATA_Y, metric="sqeuclidean")
            self.cache_cor = np.exp(-self.beta * self.cache_mol)
        
        return (self.alpha * self.cache_cor)
    
    def diff_theta(self) -> Tuple:
        diff_alpha = self.cache_cor
        diff_beta  = self.alpha * self.cache_cor * (-self.cache_mol)
        return (diff_alpha, diff_beta)
    
    def update_theta(self, Δtheta:Tuple, eta:Tuple, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:
        # L1正則化が有効か
        if self.isL1Reg:
            Δalpha = soft_threshold(self.alpha + eta[0] * Δtheta[0], eta[0] * l1_norm)
            Δbeta  = soft_threshold(self.beta  + eta[1] * Δtheta[1], eta[1] * l1_norm)
        else:
            Δalpha = self.alpha + eta[0] * Δtheta[0]
            Δbeta  = self.beta  + eta[1] * Δtheta[1]
        
        # L2正則化が有効か
        if self.isL2Reg:
            Δalpha = Δalpha / (1 + eta[0] * l2_norm)
            Δbeta  = Δbeta  / (1 + eta[1] * l2_norm)
        
        # 非負制約が有効か
        if self.isNonNeg:
            Δalpha = np.maximum(Δalpha, 0)
            Δbeta  = np.maximum(Δbeta,  0)
        
        self.alpha = Δalpha
        self.beta  = Δbeta
        return True
    
    def get_theta(self) -> Tuple:
        return (self.alpha, self.beta)
    
    def set_theta(self, thetas:Tuple) -> bool:
        self.alpha = thetas[0]
        self.beta  = thetas[1]
        return True
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

# 対数ガウスカーネル(Log-RBFカーネル)
class LogGaussKernel(Kernel):
    def __init__(self, alpha:float, beta:float, isL1Reg:bool=True, isL2Reg:bool=True, isNonNeg:bool=True):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.isL1Reg   = isL1Reg
        self.isL2Reg   = isL2Reg
        self.isNonNeg  = isNonNeg
        self.cache_mol = np.zeros([1, 1])
        self.cache_cor = np.zeros([1, 1])
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
        
        # 前処理として符号付対数変換を行う
        symmetriclog = lambda x: np.sign(x) * np.log(np.abs(x))
        if type(DATA_Y) == type(None):
            self.cache_mol = spd.squareform(spd.pdist(symmetriclog(DATA_X), metric="sqeuclidean"))
            self.cache_cor = np.exp(-self.beta * self.cache_mol)
        else:
            self.cache_mol = spd.cdist(DATA_X, DATA_Y, metric="sqeuclidean")
            self.cache_cor = np.exp(-self.beta * self.cache_mol)
        
        return (self.alpha * self.cache_cor)
    
    def diff_theta(self) -> Tuple:
        diff_alpha = self.cache_cor
        diff_beta  = self.alpha * self.cache_cor * (-self.cache_mol)
        return (diff_alpha, diff_beta)
    
    def update_theta(self, Δtheta:Tuple, eta:Tuple, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:
        # L1正則化が有効か
        if self.isL1Reg:
            Δalpha = soft_threshold(self.alpha + eta[0] * Δtheta[0], eta[0] * l1_norm)
            Δbeta  = soft_threshold(self.beta  + eta[1] * Δtheta[1], eta[1] * l1_norm)
        else:
            Δalpha = self.alpha + eta[0] * Δtheta[0]
            Δbeta  = self.beta  + eta[1] * Δtheta[1]
        
        # L2正則化が有効か
        if self.isL2Reg:
            Δalpha = Δalpha / (1 + eta[0] * l2_norm)
            Δbeta  = Δbeta  / (1 + eta[1] * l2_norm)
        
        # 非負制約が有効か
        if self.isNonNeg:
            Δalpha = np.maximum(Δalpha, 0)
            Δbeta  = np.maximum(Δbeta,  0)
        
        self.alpha = Δalpha
        self.beta  = Δbeta
        return True
    
    def get_theta(self) -> Tuple:
        return (self.alpha, self.beta)
    
    def set_theta(self, thetas:Tuple) -> bool:
        self.alpha = thetas[0]
        self.beta  = thetas[1]
        return True
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

class LinearKernel(Kernel):
    def __init__(self, alpha:float, isL1Reg:bool=True, isL2Reg:bool=True, isNonNeg:bool=True):
        super().__init__()
        self.alpha     = alpha
        self.isL1Reg   = isL1Reg
        self.isL2Reg   = isL2Reg
        self.isNonNeg  = isNonNeg
        self.cache     = np.zeros([1, 1])
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
            self.cache = DATA_X @ DATA_X.T
        else:
            self.cache = DATA_X @ DATA_Y.T
        
        return (self.alpha * self.cache)
    
    def diff_theta(self) -> Tuple:
        diff_alpha = self.cache
        return (diff_alpha,)
    
    def update_theta(self, Δtheta:Tuple, eta:Tuple, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:
        # L1正則化が有効か
        if self.isL1Reg:
            Δalpha = soft_threshold(self.alpha + eta[0] * Δtheta[0], eta[0] * l1_norm)
        else:
            Δalpha = self.alpha + eta[0] * Δtheta[0]
        
        # L2正則化が有効か
        if self.isL2Reg:
            Δalpha = Δalpha / (1 + eta[0] * l2_norm)
        
        # 非負制約が有効か
        if self.isNonNeg:
            Δalpha = np.maximum(Δalpha, 0)
        
        self.alpha = Δalpha
        return True
    
    def get_theta(self) -> Tuple:
        return (self.alpha,)
    
    def set_theta(self, thetas:Tuple) -> bool:
        self.alpha = thetas[0]
        return True
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

class ExponentialKernel(Kernel):
    def __init__(self, alpha:float, beta:float, isL1Reg:bool=True, isL2Reg:bool=True, isNonNeg:bool=True):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.isL1Reg   = isL1Reg
        self.isL2Reg   = isL2Reg
        self.isNonNeg  = isNonNeg
        self.cache_mol = np.zeros([1, 1])
        self.cache_cor = np.zeros([1, 1])
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
            self.cache_mol = spd.squareform(spd.pdist(DATA_X, metric="cityblock"))
            self.cache_cor = np.exp(-self.beta * self.cache_mol)
        else:
            self.cache_mol = spd.cdist(DATA_X, DATA_Y, metric="cityblock")
            self.cache_cor = np.exp(-self.beta * self.cache_mol)
        
        return (self.alpha * self.cache_cor)
    
    def diff_theta(self) -> Tuple:
        diff_alpha = self.cache_cor
        diff_beta  = self.alpha * self.cache_cor * (-self.cache_mol)
        return (diff_alpha, diff_beta)
    
    def update_theta(self, Δtheta:Tuple, eta:Tuple, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:
        # L1正則化が有効か
        if self.isL1Reg:
            Δalpha = soft_threshold(self.alpha + eta[0] * Δtheta[0], eta[0] * l1_norm)
            Δbeta  = soft_threshold(self.beta  + eta[1] * Δtheta[1], eta[1] * l1_norm)
        else:
            Δalpha = self.alpha + eta[0] * Δtheta[0]
            Δbeta  = self.beta  + eta[1] * Δtheta[1]
        
        # L2正則化が有効か
        if self.isL2Reg:
            Δalpha = Δalpha / (1 + eta[0] * l2_norm)
            Δbeta  = Δbeta  / (1 + eta[1] * l2_norm)
        
        # 非負制約が有効か
        if self.isNonNeg:
            Δalpha = np.maximum(Δalpha, 0)
            Δbeta  = np.maximum(Δbeta,  0)
        
        self.alpha = Δalpha
        self.beta  = Δbeta
        return True
    
    def get_theta(self) -> Tuple:
        return (self.alpha, self.beta)
    
    def set_theta(self, thetas:Tuple) -> bool:
        self.alpha = thetas[0]
        self.beta  = thetas[1]
        return True
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)

class PeriodicKernel(Kernel):
    def __init__(self, alpha:float, beta:float, gamma:float, isL1Reg:bool=True, isL2Reg:bool=True, isNonNeg:bool=True):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.isL1Reg   = isL1Reg
        self.isL2Reg   = isL2Reg
        self.isNonNeg  = isNonNeg
        self.cache_mol = np.zeros([1, 1])
        self.cache_cor = np.zeros([1, 1])
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
            self.cache_mol = spd.squareform(spd.pdist(DATA_X, metric="cityblock"))
            self.cache_cor = np.exp(self.beta * np.cos(self.gamma * self.cache_mol))
        else:
            self.cache_mol = spd.cdist(DATA_X, DATA_Y, metric="cityblock")
            self.cache_cor = np.exp(self.beta * np.cos(self.gamma * self.cache_mol))
        
        return (self.alpha * self.cache_cor)
    
    def diff_theta(self) -> Tuple:
        diff_alpha = self.cache_cor
        diff_beta  = self.alpha * self.cache_cor * np.cos(self.cache_mol)
        diff_gamma = self.alpha * self.cache_cor * self.beta * (-np.sin(self.cache_mol)) * self.cache_mol
        return (diff_alpha, diff_beta, diff_gamma)
    
    def update_theta(self, Δtheta:Tuple, eta:Tuple, l1_norm:float=0.0, l2_norm:float=0.0) -> bool:
        # L1正則化が有効か
        if self.isL1Reg:
            Δalpha = soft_threshold(self.alpha + eta[0] * Δtheta[0], eta[0] * l1_norm)
            Δbeta  = soft_threshold(self.beta  + eta[1] * Δtheta[1], eta[1] * l1_norm)
            Δgamma = soft_threshold(self.gamma + eta[2] * Δtheta[2], eta[2] * l1_norm)
        else:
            Δalpha = self.alpha + eta[0] * Δtheta[0]
            Δbeta  = self.beta  + eta[1] * Δtheta[1]
            Δgamma = self.gamma + eta[2] * Δtheta[2]
        
        # L2正則化が有効か
        if self.isL2Reg:
            Δalpha = Δalpha / (1 + eta[0] * l2_norm)
            Δbeta  = Δbeta  / (1 + eta[1] * l2_norm)
            Δgamma = Δgamma / (1 + eta[2] * l2_norm)
        
        # 非負制約が有効か
        if self.isNonNeg:
            Δalpha = np.maximum(Δalpha, 0)
            Δbeta  = np.maximum(Δbeta,  0)
            Δgamma = np.maximum(Δgamma, 0)
        
        self.alpha = Δalpha
        self.beta  = Δbeta
        self.gamma = Δgamma
        return True
    
    def get_theta(self) -> Tuple:
        return (self.alpha, self.beta, self.gamma)
    
    def set_theta(self, thetas:Tuple) -> bool:
        self.alpha = thetas[0]
        self.beta  = thetas[1]
        self.gamma = thetas[2]
        return True
    
    def __add__(self, other):
        if not isinstance(other, Kernel):
            other = ConstantKernel(other)
        return SumKernel(self, other)



class Gaussian_Process_Regression:
    def __init__(self,
                 mat_data_x:np.ndarray,              # 学習対象入力データX
                 vec_data_y:np.ndarray,              # 学習対象出力データY
                 kernel:Kernel,                      # 相関カーネル
                 norm_α:float=1.0,                   # L1・L2正則化パラメータの強さ
                 l1_ratio:float=0.1,                 # L1・L2正則化の強さ配分・比率
                 eta:float=1e-5,                     # 学習率η
                 tol:float=1e-8,                     # 許容誤差
                 max_iterate:int=300000) -> None:    # 最大ループ回数
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

        self.mat_data_x        = np.copy(mat_data_x)
        self.vec_data_y        = np.copy(vec_data_y)
        self.kernel            = kernel
        self.norm_α            = np.abs(norm_α)
        self.l1_ratio          = np.where(l1_ratio < 0, 0, np.where(l1_ratio > 1, 1, l1_ratio))
        self.eta               = eta
        self.tol               = tol
        self.max_iterate       = round(max_iterate)


    def fit(self, solver:str='external library', visible_flg:bool=False, useRAdam:bool=False) -> bool:
        # 本ライブラリにおいてエラーの出力を行わないのは、近似的にでも処理結果が欲しいためである
        # また、solverとしてISTA・FISTAを使用する際にも注意が必要である
        # ISTA・FISTAは勾配降下法に似た特徴を有しており、対象の最適化パラメータのスケールに弱い
        # 最適化対象のパラメータの解析解のスケールに依存して、必要な更新回数が多くなる
        # スケールが極端に大きい場合などには事実上収束しないが、そもそも解析解のスケールを事前に知らない・気にしていない場合も多い
        # そのような場合には、教師データ(X, Y)をそれぞれ標準化することで対処できる
        # isStandardization=True に設定しておくことを強く推奨する
        
        x_data = self.mat_data_x
        y_data = self.vec_data_y
        
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
            
        
        # 本ライブラリで実装されているアルゴリズムは以下の4点となる
        # ・sklearnライブラリに実装されているGaussianProcessRegressor(外部ライブラリ)
        # ・座標降下法アルゴリズム(CD: Coordinate Descent Algorithm)
        # ・メジャライザー最適化(ISTA: Iterative Shrinkage soft-Thresholding Algorithm)
        # ・メジャライザー最適化(ISTA: Iterative Shrinkage soft-Thresholding Algorithm)の亜種
        # これらのアルゴリズムは全て同じ目的関数を最適化している
        # しかし、実際に同一のパラメータでパラメータ探索をさせても同一の解は得られない
        # これは、実装の細かな違いによるものであったり、解析解ではなく近似解が得られるためであったりする
        # 特にISTAは勾配降下法と同等の性質を有しているため、異なる近似解が得られる
        # すなわち実行のたびに異なる解が導かれるかつ極所最適解に落ち着くことがある
        # また、外部ライブラリとしてsklearn.gaussian_process.GaussianProcessRegressorを利用することもできる
        # この外部ライブラリは内部でコレスキー分解による連立方程式という形で解を求めている点で本ライブラリと同等である
        # 一方で、この外部ライブラリはC言語(lapack)を利用してチューニングが行われている
        # また広く公開され、多くの人に利用されているライブラリでもあるため速度・品質ともにレベルが高い
        # 探索解の品質を保証したいのであれば、外部ライブラリの利用を強く推奨する
        # 一方で、外部ライブラリはL1・L2正則化及び非負制約について付加することができない点に注意が必要である
        # 最後に広く認められているわけではないため使用の際には注意が必要であるが、本ライブラリにて実装済みの
        # これら3種類のアルゴリズムが想定する目的関数は以下のとおり
        # λ_1 = 正則化の強度 * l1_ratio
        # λ_2 = 正則化の強度 * (1 - l1_ratio)
        # μ = 目的変数y の平均値
        # Σ = カーネルK(x,x')による分散共分散行列
        # math: \begin{equation}
        # math: \begin{split}
        # math: Objective &= Normal(μ_2 + Σ_21 Σ_11^-1 (x_1 - μ_1), Σ_22 - Σ_11^-1 Σ_12)
        # math: &= Normal(Σ_21 Σ_11^-1 x_1, Σ_22 - Σ_11^-1 Σ_12)
        # math: \end{split}
        # math: \end{equation}
        # 参考までに各オプションごとの実行速度は以下の通り
        # external library  >>  FISTA  >>  ISTA
        
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
                                alpha=1e-8, random_state=0)
            self.model.fit(self.mat_data_x, self.vec_data_y)
            self.solver = "external library"

            
        elif solver == "ISTA":
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)及び非負制約を行なっている
            # 注意点として、各種正則化・制約の適用有無はカーネル単位で決定されている
            # それはカーネル毎にその性質が全く異なり、一律に適用することができないためである
            # この処理部では、あくまで各種正則化・制約を適用する場合の強度λを決定するのみである
            # 実装アルゴリズムは一般的なメジャライザー最適化(ISTA: Iterative Shrinkage soft-Thresholding Algorithm)である
            # このアルゴリズムのメジャライザー部分は勾配降下法の更新式に等しい
            # このアルゴリズムを利用する際の注意点として、以下の２つが挙げられる
            # ・教師データ(X, Y)がそれぞれ標準化されている必要があること
            # ・設定イレーション回数が十分でない場合に、大域的最適解への収束が保証できないこと
            # 無条件に教師データに正規化処理を施すことにしたため、概ねの場合に問題ない
            l1_norm = self.norm_α * self.l1_ratio
            l2_norm = self.norm_α * (1 - self.l1_ratio)
            for idx in range(0, self.max_iterate):
                K     = self.kernel.correlation(x_data)
                dK    = self.kernel.diff_theta()
                ΔDiff = diff_log_likelihood(K, y_data, dK)
                ETA   = tuple(self.eta for _ in self.kernel.get_theta())
                self.kernel.update_theta(ΔDiff, ETA, l1_norm, l2_norm)
                
                mse = np.sum(elem ** 2 for elem in ΔDiff)
                if idx % 1000 == 0:
                    print(f"ite:{idx+1}  mse:{mse}  x_new:{self.kernel.get_theta()}")
                
                if np.sqrt(mse) <= self.tol:
                    break

            self.solver = "ISTA"
        
        elif solver == "OPTIMIZER":
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)及び非負制約を行なっている
            # 注意点として、各種正則化・制約の適用有無はカーネル単位で決定されている
            # それはカーネル毎にその性質が全く異なり、一律に適用することができないためである
            # この処理部では、あくまで各種正則化・制約を適用する場合の強度λを決定するのみである
            # 実装アルゴリズムはメジャライザー最適化(ISTA: Iterative Shrinkage soft-Thresholding Algorithm)の亜種である
            # 近接勾配法にオプティマイザとしてRafaelを適用した
            # 収束速度の向上と解の安定性の両面でプレーンなISTAよりも向上している
            # ただし、Rafaelというオプティマイザはアドインテ社員による自作アルゴリズムである
            # そのため、絶対的な信頼性に乏しい
            # 信頼性を重視する場合にはRAdamというオプティマイザを使用するようにしてほしい
            # このアルゴリズムのメジャライザー部分は勾配降下法の更新式に等しい
            # このアルゴリズムを利用する際の注意点として、以下の２つが挙げられる
            # ・教師データ(X, Y)がそれぞれ標準化されている必要があること
            # ・設定イレーション回数が十分でない場合に、大域的最適解への収束が保証できないこと
            # 無条件に教師データに正規化処理を施すことにしたため、概ねの場合に問題ない
            if useRAdam:
                optims = tuple(Update_RAdam(self.eta)  for _ in self.kernel.get_theta())
            else:
                optims = tuple(Update_Rafael(self.eta) for _ in self.kernel.get_theta())
            l1_norm = self.norm_α * self.l1_ratio
            l2_norm = self.norm_α * (1 - self.l1_ratio)
            for idx in range(0, self.max_iterate):
                K         = self.kernel.correlation(x_data)
                dK        = self.kernel.diff_theta()
                ΔDiff     = diff_log_likelihood(K, y_data, dK)
                ΔDiff_opt = tuple(optim.update(diff) for optim, diff in zip(optims, ΔDiff))
                ΔDiff_upd = tuple(update for _, update in ΔDiff_opt)
                ΔDiff_alp = tuple(alpha  for alpha, _  in ΔDiff_opt)
                self.kernel.update_theta(ΔDiff_upd, ΔDiff_alp, l1_norm, l2_norm)
                
                mse = np.sum(elem ** 2 for elem in ΔDiff)
                if np.isnan(mse):
                    raise

                if idx % 100 == 0:
                    # print(f"ite:{idx+1}  Abs Err:{np.abs(Base_Loss - mse)}  x_new:{self.kernel.get_theta()}")
                    print(f"ite:{idx+1}  Abs Err:{mse}  x_new:{self.kernel.get_theta()}")
                
                if np.sqrt(mse) <= self.tol:
                    break

            self.solver = "OPTIMIZER"
            
        else:
            raise

        return True
    
    def predict(self, test_X:np.ndarray, return_std:bool=False, return_cov:bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if return_std and return_cov:
            raise RuntimeError("At most one of return_std or return_cov can be requested.")

        if self.solver == "external library":
            res = self.model.predict(test_X, return_std=return_std, return_cov=return_cov)
            return res

        x_data = self.mat_data_x
        y_data = self.vec_data_y
        x_test = test_X

        # x軸の標準化
        x_data = (x_data - self.x_mean) / self.x_std_dev
        x_test = (x_test - self.x_mean) / self.x_std_dev
        # y軸の標準化
        y_data = (y_data - self.y_mean) / self.y_std_dev
        

        # 予測平均・予測分散の算出
        K      = self.kernel.correlation(x_data)
        K_     = self.kernel.correlation(x_data, x_test)
        K__    = self.kernel.correlation(x_test)
        y_pred = K_.T @ modcho_solve(K, y_data)
        y_var  = K__ - K_.T @ modcho_solve(K, K_)
        y_var  = np.maximum(y_var, 0)


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




