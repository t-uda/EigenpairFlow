import numpy as np

def calculate_magnitudes_and_pseudo(Qs, Lambdas, n, zero_indices):
    """
    追跡された固有対を用いて、マグニチュードと擬似マグニチュードを計算する。

    Args:
        Qs (list of np.ndarray): 固有ベクトル行列 Q のリスト。
        Lambdas (list of np.ndarray): 対角固有値行列 Lambda のリスト。
        n (int): 行列のサイズ。
        zero_indices (list of int): ゼロをまたぐ固有値のインデックスのリスト。

    Returns:
        tuple[list[float], list[float]]:
            - magnitudes: 計算されたマグニチュードのリスト。
            - pseudo_magnitudes: 計算された擬似マグニチュードのリスト。
    """
    magnitudes = []
    pseudo_magnitudes = []

    for i in range(len(Qs)):
        Q_t = Qs[i]
        Lambda_t = Lambdas[i]

        # Do not use pinv here because we need to track potentially diverging magnitudes. We do not take care of numerical problems inverting the close-to-zero values here.
        Lambda_inverse = np.linalg.inv(Lambda_t)


        v = Q_t.T @ np.ones(n)

        mag = v.T @ Lambda_inverse @ v
        magnitudes.append(mag)

        pseudo_Lambda_inverse = Lambda_inverse.copy()
        if zero_indices:
            # Ensure indices are within bounds
            valid_indices = [idx for idx in zero_indices if idx < pseudo_Lambda_inverse.shape[0]]
            pseudo_Lambda_inverse[valid_indices, valid_indices] = 0

        pseudo_mag = v.T @ pseudo_Lambda_inverse @ v
        pseudo_magnitudes.append(pseudo_mag)

    return magnitudes, pseudo_magnitudes
