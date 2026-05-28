import numpy as np
from numba import njit

@njit
def sum_neighbors(S, i, j, N):
    top = (i - 1) % N
    bottom =  (i + 1) % N
    left = (j - 1) % N
    right = (j + 1) % N
    return (S[top, j] + S[bottom, j] + S[i, left] + S[i, right] +
            S[top, left] + S[top, right] + S[bottom, left] + S[bottom, right])

@njit
def delta_e(S, i, j, N, J, B):
    return 2.0 * S[i, j] * (J * sum_neighbors(S, i, j, N) + B)

@njit
def total_energy(S, N, J, B):
    energy = 0.0
    for i in range(N):
        for j in range(N):
            energy += -J * S[i, j] * sum_neighbors(S, i, j, N) / 2.0
            energy -= B * S[i, j]
    return energy

@njit
def try_rotate(S, i, j, N, J, B, beta):
    dE = delta_e(S, i, j, N, J, B)
    if dE < 0 or np.random.rand() < np.exp(-beta * dE):
        S[i, j] = -S[i, j]
        return True, dE
    return False, 0.0

@njit
def macrostep(S, N, J, B, beta, current_energy, current_magnetization):
    num_trials = N * N
    for _ in range(num_trials):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        old_spin = S[i, j]
        succesful, dE = try_rotate(S, i, j, N, J, B, beta)
        
        if succesful:
            current_energy += dE
            current_magnetization += 2 * (-old_spin)
            
    return current_energy, current_magnetization

@njit
def run_simulation(N, J, B, beta, M):
    grid = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(N, N))
    
    current_energy = total_energy(grid, N, J, B)
    current_magnetization = np.sum(grid)

    grid_history = []
    energy_history = np.zeros(M)
    magnetization_history = np.zeros(M)

    for step in range(M):
        current_energy, current_magnetization = macrostep(
            grid, N, J, B, beta, current_energy, current_magnetization
        )
        
        grid_history.append(grid.copy())
        energy_history[step] = current_energy
        magnetization_history[step] = current_magnetization / (N * N)
        
    return grid_history, energy_history, magnetization_history