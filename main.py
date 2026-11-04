
import matplotlib.pyplot as plt

from analysis.plotting import plot_relative_flux_slice

from analysis.plotting import (
    plot_flux_slice,
    plot_capture_slice
)
from models.beryllium import calculate_flux_field
from models.constants import MACRO_SIGMA_CAPTURE

field = calculate_flux_field(
    radius=2.0,
    sigma=MACRO_SIGMA_CAPTURE,
    intensity=3e8,
    resolution=101
)

fig1 = plot_flux_slice(field, radius=2.0)
fig2 = plot_capture_slice(field, radius=2.0)
fig3 = plot_relative_flux_slice(field)
plt.show()
