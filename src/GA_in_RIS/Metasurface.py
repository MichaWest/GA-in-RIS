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
    def wavelength(self) -> float: 
        c = 3e8 
        return c / self.frequency

    @property
    def k0(self) -> float:
        c = 3e8
        return 2 * np.pi / self.wavelength

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
        self.wavelength = config.wavelength

        m = np.arange(self.M) - (self.M - 1) / 2
        n = np.arange(self.N) - (self.N - 1) / 2

        self.x_grid, self.y_grid = np.meshgrid(
            m * self.dx, 
            n * self.dy, 
            indexing='ij'
        )

        self.U, self.V, self.theta_grid, self.phi_grid, self.visible_mask = self._build_uv_grid()

    def calculate_far_field(self, coding_matrix: np.ndarray) -> np.ndarray:
        return self.calculate_far_field_with_source(coding_matrix, 0.0, 0.0)

    def calculate_far_field_with_source(self,
                                        coding_matrix: np.ndarray,
                                        theta_source: float,
                                        phi_source: float) -> np.ndarray:
        phase_reflection = np.where(coding_matrix == 0, np.pi, 0.0)

        u_source = np.sin(theta_source) * np.cos(phi_source)
        v_source = np.sin(theta_source) * np.sin(phi_source)

        phase_source = - self.k0 * (self.x_grid * u_source + self.y_grid * v_source)

        total_phase = phase_reflection + phase_source
        aperture_field = self.amplitude * np.exp(1j * total_phase)

        scattering = fftshift(ifft2(aperture_field)) 

        element_pattern = np.zeros_like(self.theta_grid, dtype=float)
        element_pattern[self.visible_mask] = np.cos(self.theta_grid[self.visible_mask])

        far_field = scattering * element_pattern 
        far_field[~self.visible_mask] = 0.0 

        return far_field

    def _build_uv_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fx = np.fft.fftshift(np.fft.fftfreq(self.M, d=self.dx))
        fy = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dy))

        U, V = np.meshgrid(
            self.wavelength * fx, 
            self.wavelength * fy, 
            indexing="ij"
        )

        radial_sq = U ** 2 + V ** 2 
        visible = radial_sq <= 1.0 

        theta = np.full_like(U, np.nan, dtype=float)
        theta[visible] = np.arcsin(np.sqrt(radial_sq[visible]))

        phi = np.arctan2(V, U)

        return U, V, theta, phi, visible 

    def get_uv_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.U, self.V, self.theta_grid, self.phi_grid, self.visible_mask
    
    def get_angles(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.theta_grid, self.phi_grid