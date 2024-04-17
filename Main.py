import numpy as np
import matplotlib.pyplot as plt
import time
import os
import matplotlib.animation as ani

epsilon_0 = 8.854187817e-12  # Vacuum's Electric permittivity (in C.V^{-1}.m^{-1}
L = 50  # Length of the thruster (in cm)
a = 10  # Length of a side of thruster's cross-section (in cm)
V0 = 12  # Voltage (in V)
N = 100  # Discretization of the space dimensions
Nt = 1000
x = np.linspace(0, L, N)
y = np.linspace(0, a, N)
t_obs = 10 # Observation time in seconds
h_x = (L * 1e-2) / N
h_y = (a * 1e-2) / N
h_t = t_obs / Nt

class particle:
    def __init__(self, N):
        self.position = [0, np.random.randint(0, N-1, size=2)]
        self.velocity = []
        self.acceleration = []

class Cation(particle):
    def __init__(self, N):
        super().__init__(N)
        self.charge = 1.602e-19

class Anion(particle):
    def __init__(self, N):
        super().__init__(N)
        self.charge = -1.602e-19

def solve_poisson(particles, plot=False):
    V = np.zeros((N, N))
    Vprime = np.zeros((N, N))
    rho = np.zeros((N, N))
    V[0, :] = V0 * np.sin(2 * np.pi * x / L)
    V[-1, :] = -V0 * np.cos(2 * np.pi * x / L)
    V[:, 0] = 0
    V[:, -1] = 0
    for part in particles:
        rho[part.position[0], part.position[1]] = part.charge

    iteration = 0
    delta = 1
    max_iter = 10000
    tol = 1e-4

    while delta > tol and iteration < max_iter:
        print("Iteration", iteration)
        coef_x = (h_y ** 2) / (h_x ** 2 + h_y ** 2)
        coef_y = (h_x ** 2) / (h_x ** 2 + h_y ** 2)
        coef_rho = ((h_x ** 2) * (h_y ** 2)) / (2 * (h_x ** 2 + h_y ** 2) * epsilon_0)
        Vprime[1:N - 1, 1:N - 1] = (0.5 * (coef_x * (V[2:N, 1:N - 1] + V[0:N - 2, 1:N - 1]) +
                                          coef_y * (V[1:N - 1, 2:N] + V[1:N - 1, 0:N - 2])) +
                                    coef_rho * rho[1:N - 1, 1:N - 1])
        Vprime[0, :] = V0 * np.sin(2 * np.pi * x / L)
        Vprime[-1, :] = -V0 * np.cos(2 * np.pi * x / L)
        Vprime[:, 0] = 0
        Vprime[:, -1] = 0

        delta = np.max(np.abs(V - Vprime))
        V, Vprime = Vprime, V

        print("Delta", delta)
        iteration += 1

    if plot:
        fig1 = plt.figure()
        potential = fig1.add_subplot(111)

        potential.set_title('Potential in space')
        potential.set_xlabel('x (in cm)')
        potential.set_ylabel('y (in cm)')

        im = potential.imshow(V, extent=(0, L, 0, a))
        plt.colorbar(im)

        fig2 = plt.figure()
        rho_viewer = fig2.add_subplot(111)

        rho_viewer.set_title('Charge distribution in space')
        rho_viewer.set_xlabel('x (in cm)')
        rho_viewer.set_ylabel('y (in cm)')

        im2 = rho_viewer.imshow(rho, extent=(0, L, 0, a))
        plt.colorbar(im2)
        plt.show()
    return V

def calculate_E(V, plot=False):
    E = np.zeros_like(V)
    E[:, 1:-1] = - (V[:, 2:] - V[:, :-2]) / (2 * h_x)
    E[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 * h_y)

    if plot:
        fig = plt.figure()
        e_viewer = fig.add_subplot(111)
        e_viewer.set_title('Electric field in space')
        e_viewer.set_xlabel('x (in cm)')
        e_viewer.set_ylabel('y (in cm)')

        im = e_viewer.imshow(E, extent=(0, L, 0, a))
        plt.colorbar(im)
        plt.show()
    return E

def MaxwellFaraday(B, E, plot=False):
    newB = np.zeros_like(E)
    newB[1:N, :N - 1] = B[1:N, :N - 1] - (h_t / h_x) * (E[1:N, :N - 1] - E[:N-1, :N - 1]) + (h_t / h_y) * (E[1:N, 1:N] - E[:N-1, :N - 1])

    if plot:
        fig = plt.figure()
        e_viewer = fig.add_subplot(111)
        e_viewer.set_title('Electric field in space')
        e_viewer.set_xlabel('x (in cm)')
        e_viewer.set_ylabel('y (in cm)')

        im = e_viewer.imshow(newB, extent=(0, L, 0, a))
        plt.colorbar(im)
        plt.show()

B = np.ones((N, N))
particles = [Cation(N) for i in range(10)]
pot = solve_poisson(particles)
E = calculate_E(pot)
MaxwellFaraday(B, E, plot=True)

