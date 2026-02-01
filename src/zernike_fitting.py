# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 15:04
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : lsi-algorithm
@File        : zernike_fitting.py
@Software    : PyCharm
@Description : 
"""

import numpy as np
from scipy.linalg import lstsq
from fringe_zernike_generator import FringeZernike  # 导入用户的Zernike生成类


class ZernikeFitter:
    """Zernike拟合类：最小二乘拟合相位（适配用户Fringe索引Zernike）"""

    def __init__(self,
                 unwrapped_phase: np.ndarray,
                 rho: np.ndarray,
                 theta: np.ndarray,
                 max_order: int,  # Fringe最大索引
                 circle_mask: np.ndarray):
        """
        初始化参数（适配Fringe索引）
        :param unwrapped_phase: 解包裹相位（圆外NaN）
        :param rho: 极径矩阵
        :param theta: 极角矩阵
        :param max_order: Fringe索引最大阶数
        :param circle_mask: 内切圆掩码
        """
        self.unwrapped_phase = unwrapped_phase
        self.rho = rho
        self.theta = theta
        self.max_order = max_order
        self.circle_mask = circle_mask

        # 拟合结果
        self.fitted_coeffs = None
        self.fitted_phase = None
        self.zernike_generator = None  # 用户的FringeZernike实例

    def fit(self) -> tuple:
        """最小二乘拟合（核心方法：使用用户的Zernike生成基底）"""
        # 1. 筛选有效像素：圆内+非NaN
        valid_mask = self.circle_mask & (~np.isnan(self.unwrapped_phase))
        if np.sum(valid_mask) == 0:
            raise ValueError("无有效像素用于拟合！")
        K = np.sum(valid_mask)  # 有效像素数
        print(f"✅ 拟合有效像素数 = {K}")

        # 2. 初始化用户的FringeZernike生成器（分辨率与相位矩阵一致）
        resolution = self.unwrapped_phase.shape[0]
        self.zernike_generator = FringeZernike(
            max_order=self.max_order,
            resolution=resolution
        )
        # 替换用户生成器的网格为当前极坐标（关键修复）
        self.zernike_generator.rr = self.rho  # 原始极径
        self.zernike_generator.tt = self.theta  # 原始极角
        self.zernike_generator.x = self.rho * np.cos(self.theta)
        self.zernike_generator.y = self.rho * np.sin(self.theta)

        # 3. 构建Zernike基底矩阵A（K×max_order）
        A = np.zeros((K, self.max_order), dtype=np.float64)
        for idx in range(1, self.max_order + 1):
            # 用用户的generate方法生成对应Fringe索引的Zernike多项式
            zernike_poly = self.zernike_generator.generate(idx)
            # 仅提取圆形区域内的有效像素
            A[:, idx - 1] = zernike_poly[valid_mask]

        # 4. 最小二乘拟合
        y = self.unwrapped_phase[valid_mask]
        self.fitted_coeffs, _, _, _ = lstsq(A, y, cond=None)

        # 5. 生成拟合相位（仅圆形区域有效，圆外NaN）
        self.fitted_phase = np.full_like(self.unwrapped_phase, np.nan, dtype=np.float64)
        fitted_phase_valid = A @ self.fitted_coeffs  # 矩阵乘法加速
        self.fitted_phase[valid_mask] = fitted_phase_valid

        return self.fitted_coeffs, self.fitted_phase
