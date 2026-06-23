from typing import Tuple

import numpy as np
from dataclasses import dataclass

from numpy.fft import ifft2, ifftshift, fftshift


@dataclass
class MetasurfaceConfig:
    M: int = 40
    N: int = 40
    dx: float = 5.8e-3
    dy: float = 4.9e-3
    frequency: float = 11.1e9
    amplitude_reflection: float = 0.95

    @property
    def k0(self) -> float:
        c = 3e8
        return 2 * np.pi * self.frequency / c

class UnitCell:
    @staticmethod
    def get_phase(state: int, polarization: str = 'x') -> float:
        if polarization == 'x':
            if state == 0:
                return np.pi
            else:
                return 0.0
        else:
            if state == 0:
                return 0.0
            else:
                return 0.0

    @staticmethod
    def get_complex_reflection(state: int, polarization: str = 'x') -> complex:
        amplitude = 0.95  # Почти полное отражение
        phase = UnitCell.get_phase(state, polarization)
        return amplitude * np.exp(1j * phase)

class ScatteringCalculator:
    def __init__(self, config: MetasurfaceConfig):
        self.config = config
        self.M = config.M
        self.N = config.N
        self.dx = config.dx
        self.dy = config.dy
        self.k0 = config.k0
        self.amplitude = config.amplitude_reflection

        self.m_indices, self.n_indices = np.meshgrid(
            np.arange(self.M),
            np.arange(self.N),
            indexing='ij'
        )

    def calculate_far_field(self, coding_matrix: np.ndarray) -> np.ndarray:
        return self.calculate_far_field_with_source(coding_matrix, 0.0, 0.0)

    def calculate_far_field_with_source(self,
                                        coding_matrix: np.ndarray,
                                        theta_source: float,
                                        phi_source: float) -> np.ndarray:
        phase_reflection = np.where(coding_matrix == 0, np.pi, 0.0)

        sin_theta_src = np.sin(theta_source)
        cos_phi_src = np.cos(phi_source)
        sin_phi_src = np.sin(phi_source)

        phase_source = self.k0 * (
                self.m_indices * self.dx * sin_theta_src * cos_phi_src +
                self.n_indices * self.dy * sin_theta_src * sin_phi_src
        )

        total_phase = phase_reflection + phase_source
        reflection_matrix = self.amplitude * np.exp(1j * total_phase)

        scattering = fftshift(ifft2(ifftshift(reflection_matrix)))

        theta, phi = self.get_angles()
        cos_theta = np.cos(theta)

        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)

        uv_sum_sq = u ** 2 + v ** 2
        uv_sum_sq = np.clip(uv_sum_sq, 0, 1)

        denominator = np.sqrt(1 - uv_sum_sq)
        denominator[denominator < 1e-10] = 1e-10

        far_field = scattering * cos_theta / denominator

        return far_field


    def get_angles(self) -> Tuple[np.ndarray, np.ndarray]:
        M, N = self.M, self.N

        kx = np.arange(-M // 2, M // 2)
        ky = np.arange(-N // 2, N // 2)
        Kx, Ky = np.meshgrid(kx, ky)

        u = Kx / (self.k0 * self.dx)
        v = Ky / (self.k0 * self.dy)

        r = np.sqrt(u ** 2 + v ** 2)
        r = np.clip(r, 0, 1)

        theta = np.arcsin(r)
        phi = np.arctan2(v, u)

        return theta, phi