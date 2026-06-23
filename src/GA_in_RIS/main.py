import numpy as np
import matplotlib.pyplot as plt

from src.GA_in_RIS.Metasurface import MetasurfaceConfig
from src.GA_in_RIS.MetasurfaceDesigner import MetasurfaceDesigner

def visualize_coding_matrix(coding_matrix: np.ndarray, title: str = "Матрица кодирования метаповерхности"):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(coding_matrix, cmap='bwr', aspect='equal')

    cbar = plt.colorbar(im)
    cbar.set_label('State of Unit Cell', rotation=270, labelpad=20)

    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([r'$\pi$ (State 0)', r'$0$ (State 1)'])

    ax.set_title(title, fontsize=14)

    ax.set_xlabel('Cell Index N', fontsize=12)
    ax.set_ylabel('Cell Index M', fontsize=12)

    plt.tight_layout()
    plt.show()

config = MetasurfaceConfig(M=40, N=40)
designer = MetasurfaceDesigner(config)

receivers = [
        {'theta': 0.7854, 'phi': 0, 'weight': 1.0},
        {'theta': -0.7854, 'phi': 0, 'weight': 1.0}
    ]

matrix, history = designer.design_beam_steering(
    theta_source=0, phi_source=0,
    receivers=receivers
)

visualize_coding_matrix(matrix, "Закодированная метаповерхность")
designer.visualize_pattern_comparison(matrix, receivers)