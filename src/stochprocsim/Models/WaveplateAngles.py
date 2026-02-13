import numpy as np
import sympy as sp

Φm1, Φm2, Φm3 = sp.symbols('Φm1 Φm2 Φm3')
θh1, θq1, θh2, θq2, θhin2, θqin2, θhf1, θqf1, θhf2, θqf2, pipi = sp.symbols('θh1 θq1 θh2 θq2 θhin2 θqin2 θhf1 θqf1 θhf2 θqf2 pipi')

# Top angles are original, underneaths are reoptimised using finder shuffled to NTU defs

### GU
U_3_angles_GU = {
    "θh1": np.rad2deg(1.57158),
    "θq1": np.rad2deg(0.0017101),
    "θh2": np.rad2deg(3.99667),
    "θq2": np.rad2deg(1.92368),
    "θhin2": np.rad2deg(1.58668),
    "θqin2": np.rad2deg(4.79562),
    "θhf1": np.rad2deg(4.31667),
    "θqf1": np.rad2deg(4.55094),
    "θhf2": np.rad2deg(5.83508),
    "θqf2": np.rad2deg(1.38684),
    "pipi": 2.03774,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74,
}

    
U_4_angles_GU = {
    "θh1": np.rad2deg(3.144),
    "θq1": np.rad2deg(0.0),
    "θh2": np.rad2deg(0.845031),
    "θq2": np.rad2deg(2.74251),
    "θhin2": np.rad2deg(0.710524),
    "θqin2": np.rad2deg(0.0),
    "θhf1": np.rad2deg(2.54209),
    "θqf1": np.rad2deg(4.49778),
    "θhf2": np.rad2deg(5.83761),
    "θqf2": np.rad2deg(3.40202),
    "pipi": 0.270094,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74
}

U_5_angles_GU = {
    "θh1": np.rad2deg(1.5707),
    "θq1": np.rad2deg(1.57092),
    "θh2": np.rad2deg(2.48831),
    "θq2": np.rad2deg(4.44748),
    "θhin2": np.rad2deg(2.751),
    "θqin2": np.rad2deg(1.2807),
    "θhf1": np.rad2deg(4.1716),
    "θqf1": np.rad2deg(2.91341),
    "θhf2": np.rad2deg(2.83674),
    "θqf2": np.rad2deg(4.93841),
    "pipi": 3.97891,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74,
}

U_6_angles_GU = {
    "θh1": np.rad2deg(1.57076),
    "θq1": np.rad2deg(4.71234),
    "θh2": np.rad2deg(5.99134),
    "θq2": np.rad2deg(3.60292),
    "θhin2": np.rad2deg(3.1633),
    "θqin2": np.rad2deg(2.21049),
    "θhf1": np.rad2deg(4.61802),
    "θqf1": np.rad2deg(2.06573),
    "θhf2": np.rad2deg(5.65754),
    "θqf2": np.rad2deg(5.72802),
    "pipi": 0.192194,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74
}


### NTU
U_3_angles_NTU = {
    "θh1": np.rad2deg(1.57158),
    "θq1": np.rad2deg(0.0017101),
    "θh2": np.rad2deg(3.99667),
    "θq2": np.rad2deg(1.92368),
    "θhin2": np.rad2deg(1.58668),
    "θqin2": np.rad2deg(4.79562),
    "θhf1": np.rad2deg(4.31667),
    "θqf1": np.rad2deg(4.55094),
    "θhf2": np.rad2deg(5.83508),
    "θqf2": np.rad2deg(1.38684),
    "pipi": 2.03774,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74,
}

U_4_angles_NTU = {
    "θh1": np.rad2deg(3.144),
    "θq1": np.rad2deg(0.0),
    "θh2": np.rad2deg(0.845031),
    "θq2": np.rad2deg(2.74251),
    "θhin2": np.rad2deg(0.710524),
    "θqin2": np.rad2deg(0.0),
    "θhf1": np.rad2deg(2.54209),
    "θqf1": np.rad2deg(4.49778),
    "θhf2": np.rad2deg(5.83761),
    "θqf2": np.rad2deg(3.40202),
    "pipi": 0.270094,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74
}

U_5_angles_NTU = {
    "θh1": np.rad2deg(1.5707),
    "θq1": np.rad2deg(1.57092),
    "θh2": np.rad2deg(2.48831),
    "θq2": np.rad2deg(4.44748),
    "θhin2": np.rad2deg(2.751),
    "θqin2": np.rad2deg(1.2807),
    "θhf1": np.rad2deg(4.1716),
    "θqf1": np.rad2deg(2.91341),
    "θhf2": np.rad2deg(2.83674),
    "θqf2": np.rad2deg(4.93841),
    "pipi": 3.97891,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74,
}


U_6_angles_NTU = {
    "θh1": np.rad2deg(1.57076),
    "θq1": np.rad2deg(4.71234),
    "θh2": np.rad2deg(5.99134),
    "θq2": np.rad2deg(3.60292),
    "θhin2": np.rad2deg(3.1633),
    "θqin2": np.rad2deg(2.21049),
    "θhf1": np.rad2deg(4.61802),
    "θqf1": np.rad2deg(2.06573),
    "θhf2": np.rad2deg(5.65754),
    "θqf2": np.rad2deg(5.72802),
    "pipi": 0.192194,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74
}
