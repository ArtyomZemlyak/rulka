"""
Map cycle configuration.
See docs/source/configuration_guide.rst for detailed documentation.
"""

from itertools import repeat

# Nadeo maps selection
nadeo_maps_to_train_and_test = [
    "A01-Race",
    "A03-Race",
    "A05-Race",
    "A07-Race",
    "A11-Race",
    "A14-Race",
    "A15-Speed",
    "B01-Race",
    "B02-Race",
    "B03-Race",
    "B05-Race",
    "B10-Speed",
    "B14-Speed",
]

# Map cycle
# Format: (short_name, map_path, reference_line_path, is_exploration, fill_buffer)
# See documentation for detailed explanation

map_cycle = []

map_cycle += [
    # Primary: ESL-Hockolicious
    repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", True, True), 4),
    
    # Secondary: A01-Race
    repeat(("A01", '"A01-Race.Challenge.Gbx"', "A01_0.5m_cl.npy", True, True), 4),
    repeat(("A01", '"A01-Race.Challenge.Gbx"', "A01_0.5m_cl.npy", False, True), 1),
]
