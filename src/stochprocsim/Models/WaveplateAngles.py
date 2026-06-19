import numpy as np
import sympy as sp

Φm1, Φm2, Φm3 = sp.symbols('Φm1 Φm2 Φm3')
θh1, θq1, θh2, θq2, θhin2, θqin2, θhf1, θqf1, θhf2, θqf2, pipi = sp.symbols('θh1 θq1 θh2 θq2 θhin2 θqin2 θhf1 θqf1 θhf2 θqf2 pipi')

OA_angles = {
    "θh1": 71.84,
    "θq1": 179.91,
    "θh2": 195.167,
    "θq2": 155.916,
    "θhin2": 0,
    "θqin2": 0,
    "θhf1": 0,
    "θqf1": 0,
    "θhf2": 126,
    "θqf2": 144.167,
    "pipi": 0,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74,
}

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

angs = {
    3: U_3_angles_NTU, 
    4: U_4_angles_NTU, 
    5: U_5_angles_NTU, 
    6: U_6_angles_NTU
} 

def print_adj_angles(unitary_angles):
    angles_adj = {}
    for key in OA_angles.keys():
        angles_adj[key] = OA_angles[key] + unitary_angles[key]
    for key, value in angles_adj.items():
        print(f'{key}: {value%360:.2f}°')


if __name__ == "__main__":
    n = input("What unitary number?\t")
    print_adj_angles(angs[int(n)])