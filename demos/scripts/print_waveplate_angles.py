from stochprocsim.models.waveplate_angles import angs

OA_angles = {
    "θh1": 71.84,
    "θq1": 179.91,
    "θh2": 195.167,
    "θq2": 155.916,
    "θhin2": 0,
    "θqin2": 0,
    "θhf1": 40.83,
    "θqf1": 222.25,
    "θhf2": 0,
    "θqf2": 0,
    "pipi": 0,
    "Φm1": 3.16,
    "Φm2": 3.77,
    "Φm3": 3.74,
}

def print_adj_angles(unitary_angles):
    angles_adj = {k: OA_angles[k] + unitary_angles[k] for k in OA_angles}
    for key, value in angles_adj.items():
        print(f'{key}: {value % 360:.2f}°')

if __name__ == "__main__":
    n = input("What unitary number?\t")
    print_adj_angles(angs[int(n)])
