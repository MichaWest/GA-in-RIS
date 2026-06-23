from typing import Tuple, List, Dict

import numpy as np

from src.GA_in_RIS.GeneticAlgorithm import GeneticAlgorithmOptimizer
from src.GA_in_RIS.Metasurface import MetasurfaceConfig, ScatteringCalculator


class MetasurfaceDesigner:
    def __init__(self,
                 config: MetasurfaceConfig):
        self.config = config
        self.calculator = ScatteringCalculator(config)
        self.optimizer = GeneticAlgorithmOptimizer(config)

    def design_beam_steering(self,
                             theta_source: float,
                             phi_source: float,
                             receivers: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:

        def fitness_function(coding_matrix: np.ndarray) -> float:
            scattering = self.calculator.calculate_far_field_with_source(
                coding_matrix,
                theta_source,
                phi_source)

            intensity = np.abs(scattering)**2
            P_total = np.sum(intensity)

            P_receivers = 0.0
            for rec in receivers:
                theta = rec['theta']
                phi = rec['phi']
                weight= rec.get('weight', 1.0)

                idx = self.__find_angle_index(theta, phi)
                P_receivers += weight * intensity.flat[idx]

            return - P_receivers / (P_total + 1e-10)

        best_matrix, history = self.optimizer.optimize(fitness_function)

        return best_matrix, history

    def __find_angle_index(self, theta: float, phi: float) -> int:
        theta_grid, phi_grid = self.calculator.get_angles()

        u_target = np.sin(theta) * np.cos(phi)
        v_target = np.sin(theta) * np.sin(phi)

        u_grid = np.sin(theta_grid) * np.cos(phi_grid)
        v_grid = np.sin(theta_grid) * np.sin(phi_grid)

        distances = (u_grid - u_target) ** 2 + (v_grid - v_target) ** 2
        return int(np.argmin(distances))

    def visualize_pattern_comparison(self,
                                     coding_matrix: np.ndarray,
                                     receivers: List[Dict[str, float]],
                                     theta_source: float = 0,
                                     phi_source: float = 0,
                                     grid_size: int = 100) -> None:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        scattering = self.calculator.calculate_far_field_with_source(
            coding_matrix,
            theta_source,
            phi_source
        )

        theta, phi = self.calculator.get_angles()

        F_real = np.abs(scattering)
        F_real_norm = F_real / np.max(F_real)

        F_desired = np.zeros_like(F_real)
        for rec in receivers:
            theta_r = rec['theta']
            phi_r = rec['phi']
            weight = rec.get('weight', 1.0)

            idx = self.__find_angle_index(theta_r, phi_r)
            F_desired.flat[idx] = weight

        F_desired_norm = F_desired / np.max(F_desired)

        U = np.sin(theta) * np.cos(phi)
        V = np.sin(theta) * np.sin(phi)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Ожидаемая ДН', 'Полученная ДН'),
            specs=[[{'type': 'surface'}, {'type': 'surface'}]]
        )

        fig.add_trace(go.Surface(
            x=U, y=V, z=F_desired_norm,
            colorscale='Viridis', showscale=False
        ), row=1, col=1)

        fig.add_trace(go.Surface(
            x=U, y=V, z=F_real_norm,
            colorscale='Plasma', showscale=False
        ), row=1, col=2)

        fig.update_layout(
            title='Сравнение ожидаемой и полученной ДН (3D)',
            width=1200, height=500,
            scene=dict(
                xaxis_title='u',
                yaxis_title='v',
                zaxis_title='|F|',
                zaxis=dict(range=[0, 1])
            ),
            scene2=dict(
                xaxis_title='u',
                yaxis_title='v',
                zaxis_title='|F|',
                zaxis=dict(range=[0, 1])
            )
        )

        fig.show()