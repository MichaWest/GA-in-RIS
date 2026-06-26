from typing import Tuple

import numpy as np
from dataclasses import dataclass

from numpy.fft import ifft2, ifftshift, fftshift


@dataclass
class MetasurfaceConfig:
    M: int = 40
    N: int = 40
    frequency: float = 5.25 * 1e9
    amplitude_reflection: float = 0.95
    @property
    def wavelength(self) -> float: 
        c = 3e8 
        return c / self.frequency

    @property
    def k0(self) -> float:
        c = 3e8
        return 2 * np.pi / self.wavelength

    @property
    def dx(self) -> float:
        return 0.45 * self.wavelength

    @property
    def dy(self) -> float:
        return 0.45 * self.wavelength

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

    def calculate_directivity(self,
                              far_field) -> float:
        intensity = np.abs(far_field) ** 2
        intensity[~self.visible_mask] = 0.0

        theta_visible = self.theta_grid[self.visible_mask]
        intensity_visible = intensity[self.visible_mask]

        dU = np.abs(self.U[1, 0] - self.U[0, 0])
        dV = np.abs(self.V[0, 1] - self.V[0, 0])

        cos_theta = np.cos(theta_visible)
        cos_theta = np.maximum(cos_theta, 1e-12)

        weights = (dU * dV) / cos_theta

        P_rad = np.sum(intensity_visible * weights)

        I_max = np.max(intensity)
        directivity = 4.0 * np.pi * I_max / P_rad if P_rad > 1e-12 else 0.0

        return 10 * np.log10(directivity)


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