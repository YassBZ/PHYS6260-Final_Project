import numpy as np
import matplotlib.pyplot as plt
import time
import os
import matplotlib.animation as ani

epsilon_0 = 80 * 8.854187817e-12  # Vacuum's Electric permittivity (in C.V^{-1}.m^{-1}
c = 299792458.0
L = 50  # Length of the thruster (in cm)
a = 10  # Length of a side of thruster's cross-section (in cm)
V0 = 12  # Voltage (in V)
N = 100  # Discretization of the space dimensions
Nt = 100
x = np.linspace(0, L, N)
y = np.linspace(0, a, N)
t_obs = 5  # Observation time in seconds
T_current = 1 / 60
h_x = (L * 1e-2) / N
h_y = (a * 1e-2) / N
h_t = t_obs / Nt

class Data:
    def __init__(self, a, L, t_obs, N, Nt, V0):
        self.a = a
        self.L = L
        self.t_obs = t_obs
        self.N = N
        self.Nt = Nt
        self.dx = (L * 1e-2) / N
        self.dy = (a * 1e-2) / N
        self.dt = t_obs / Nt
        self.figures = []
        self.particles = []
        self.time_step = 0
        self.rho = np.zeros((Nt+1, N+1, N+1))
        self.V = np.zeros((Nt+1, N+1, N+1))
        self.Ex = np.zeros((Nt+1, N+1, N+1))
        self.Ey = np.zeros((Nt+1, N+1, N+1))
        self.B = np.zeros((Nt+1, N+1, N+1))
        self.x = np.linspace(0, L * 1e-2, N+1)
        self.y = np.linspace(0, a * 1e-2, N+1)
        self.meshgrid = np.meshgrid(self.x, self.y)
        self.time = np.linspace(0, t_obs, Nt+1)
        self.V0 = V0


    def init_potential(self):
        potential = np.zeros((self.N + 1, self.N + 1), dtype=float)
        potential[0, :] = self.V0 * np.sin(2 * np.pi * ((self.x / self.L) - (self.time_step * self.dt / t_obs)))
        potential[-1, :] = self.V0 * np.sin(2 * np.pi * ((self.x / self.L) - (self.time_step * self.dt / t_obs)))
        for p in self.particles:
            if not p.out:
                potential_particle = p.update_potential(self.meshgrid[0], self.meshgrid[1], epsilon_0, self.time_step)
                potential += potential_particle
        potential[:, 0] = 0
        potential[:, -1] = 0
        self.V[self.time_step] = potential

    def generate_cations(self, N_part, time_step=0):
        for _ in range(N_part):
            particle = Cation(self.N+1, self.Nt, time_step=time_step)
            self.particles.append(particle)
            x = int(particle.position[0][0] / self.dx)
            y = int(particle.position[0][1] / self.dy)
            self.rho[self.time_step][x, y] = particle.charge

    def generate_anions(self, N_part, time_step=0):
        for _ in range(N_part):
            particle = Anion(self.N+1, self.Nt, time_step=time_step)
            self.particles.append(particle)
            x = int(particle.position[0][0] / self.dx)
            y = int(particle.position[0][1] / self.dy)
            self.rho[self.time_step][x, y] = particle.charge


    def generate_magnets(self, Nmagnets):
        B = np.zeros((self.N+1, self.N+1))
        sizemagnet = L * 1e-2 / Nmagnets
        span = int(sizemagnet / self.dx)
        for k in range(1, Nmagnets+1):
            B[:, int((k-1)*span):int(k*span)] = 1 * (-1) ** (k+1)

        self.B[:] = B

    def solve_poisson(self, plot=False):
        self.init_potential()
        Vprime = np.empty((self.N+1, self.N+1), dtype=float)
        V = self.V[self.time_step]
        iteration = 0
        delta = 1
        max_iter = 10000
        tol = 1e-4

        while delta > tol and iteration < max_iter:
            Vprime[1:N, 1:N] = 0.25 * (V[2:, 1:self.N] + V[: self.N-1, 1:self.N] + V[1:self.N, 2:] + V[1:self.N, :self.N-1])
            Vprime[0, :] = self.V0 * np.sin(2 * np.pi * ((self.x / self.L) - (self.time_step * self.dt / t_obs)))
            Vprime[-1, :] = self.V0 * np.sin(2 * np.pi * ((self.x / self.L) - (self.time_step * self.dt / t_obs)))
            for p in self.particles:
                if not p.out:
                    potential_particle = p.update_potential(self.meshgrid[0], self.meshgrid[1], epsilon_0, self.time_step)
                    Vprime += potential_particle
            Vprime[:, 0] = 0
            Vprime[:, -1] = 0
            delta = np.max(np.abs(V - Vprime))
            V, Vprime = Vprime, V

            iteration += 1
        self.V[self.time_step] = V
        print("Potential Solved")

        if plot:
            fig1 = plt.figure()
            potential = fig1.add_subplot(111)

            potential.set_title(f'Potential in space at t={self.time_step * self.dt}')
            potential.set_xlabel('x (in cm)')
            potential.set_ylabel('y (in cm)')

            im = potential.imshow(self.V[self.time_step], extent=(0, L, 0, a), origin='lower')
            plt.colorbar(im)
            for p in self.particles:
                if not p.out:
                    #potential.scatter(p.position[self.time_step][0] * 100, p.position[self.time_step][1] * 100, marker='o', s=10, c='red')
                    print(p.position[self.time_step] * 100)
            self.figures.append(fig1)

    def calculate_E(self, plot=False):
        Ex = np.zeros((self.N+1, self.N+1))
        Ey = np.zeros((self.N+1, self.N+1))

        Ex[:, 1:-1] = -(self.V[self.time_step][:, 2:] - self.V[self.time_step][:, :-2]) / (2 * self.dx)
        Ey[1:-1, :] = -(self.V[self.time_step][2:, :] - self.V[self.time_step][:-2, :]) / (2 * self.dy)

        self.Ex[self.time_step] = Ex
        self.Ey[self.time_step] = Ey

        if plot:
            fig1 = plt.figure()
            ex_viewer = fig1.add_subplot(111)
            ex_viewer.set_title(f'$E_x$ field in space at t={self.time_step * self.dt}')
            ex_viewer.set_xlabel('x (in cm)')
            ex_viewer.set_ylabel('y (in cm)')

            im1 = ex_viewer.imshow(self.Ex[self.time_step], extent=(0, L, 0, a), origin='lower')
            plt.colorbar(im1)

            fig2 = plt.figure()
            ey_viewer = fig2.add_subplot(111)
            ey_viewer.set_title(f'$E_y$ field in space at t={self.time_step * self.dt}')
            ey_viewer.set_xlabel('x (in cm)')
            ey_viewer.set_ylabel('y (in cm)')

            im2 = ey_viewer.imshow(self.Ey[self.time_step], extent=(0, L, 0, a), origin='lower')
            plt.colorbar(im2)
            self.figures.append(fig1)

    def MaxwellFaraday(self, plot=False):
        if self.time_step > 0:
            Bnew = np.zeros((self.N+1, self.N+1))
            Bnew += self.B[self.time_step - 1]
            Bnew[0, :] = 0
            Bnew[self.N, :] = 0
            Bnew[:, 0] = 0
            Bnew[:, self.N ] = 0
            dtdx = self.dt / self.dx
            dtdy = self.dt / self.dy
            Ex = self.Ex[self.time_step - 1]
            Ey = self.Ey[self.time_step - 1]

            Bnew[1:self.N, 1:self.N] += - dtdx * (Ey[1:self.N, 1:self.N] - Ey[1:self.N, :self.N-1]) + dtdy * (Ex[2:, 1:self.N] - Ex[1:self.N, 1:self.N])

            self.B[self.time_step] += Bnew

        if plot:
            fig = plt.figure()
            e_viewer = fig.add_subplot(111)
            e_viewer.set_title(f'Magnetic field in space at t={self.time_step * self.dt}')
            e_viewer.set_xlabel('x (in cm)')
            e_viewer.set_ylabel('y (in cm)')

            im = e_viewer.imshow(self.B[self.time_step], extent=(0, L, 0, a), origin='lower')
            plt.colorbar(im)
            self.figures.append(fig)

    def MoveParticles(self):
        for part in self.particles:
            x = part.position[self.time_step][0]
            y = part.position[self.time_step][1]
            vx = part.velocity[self.time_step][0]
            vy = part.velocity[self.time_step][1]
            if not part.out:
                Ex = self.Ex[self.time_step]
                Ey = self.Ey[self.time_step]
                B = self.B[self.time_step]
                index_x = int((x / self.dx)) % self.N
                index_y = int((y / self.dy)) % self.N

                new_vx = vx + (self.dt * part.charge / part.mass) * (Ex[index_x, index_y] + vy * B[index_x, index_y]) * 1e-2
                new_vy = vy + (self.dt * part.charge / part.mass) * (Ey[index_x, index_y] - vx * B[index_x, index_y]) * 1e-2
                print(x, y, vx, vy, Ex[index_x, index_y], Ey[index_x, index_y], B[index_x, index_y], new_vx, new_vy)
                newpos_X = x + self.dt * vx * 1e-2
                newpos_Y = y + self.dt * vy * 1e-2
                newpos_X = newpos_X % (L * 1e-2)
                newpos_Y = newpos_Y % (a * 1e-2)
                print(newpos_X, newpos_Y)

                if newpos_X == 0:
                    newpos_X = 2 * self.dx

                if new_vx > 1000000:
                    new_vx = 1000000
                if new_vy > 1000000:
                    new_vy = 1000000


                part.position[self.time_step + 1][0] = newpos_X
                part.position[self.time_step + 1][1] = newpos_Y

                part.velocity[self.time_step + 1][0] = new_vx
                part.velocity[self.time_step + 1][1] = new_vy
            else:
                part.position[self.time_step + 1][0] = x
                part.position[self.time_step + 1][1] = y
                part.velocity[self.time_step + 1][0] = vx
                part.velocity[self.time_step + 1][1] = vy

    def run(self, Ncations, Nanions, Nmagnets):
        self.generate_cations(Ncations, time_step=self.Nt)
        self.generate_anions(Nanions, time_step=self.Nt)
        for i in range(self.Nt):
            print("Time Step :", i)
            if i == 0 or i == self.Nt-1:
                self.generate_magnets(Nmagnets)
                self.solve_poisson(plot=True)
                self.calculate_E(plot=True)
                self.MaxwellFaraday(plot=True)
                self.MoveParticles()
                self.time_step += 1
            else:
                self.generate_magnets(Nmagnets)
                self.solve_poisson()
                self.calculate_E()
                self.MaxwellFaraday()
                self.MoveParticles()
                self.time_step += 1
        particle = self.particles[0]
        X = [particle.position[k][0] for k in range(self.Nt+1)]
        VX = [particle.velocity[k][0] for k in range(self.Nt+1)]
        time = self.time

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(time, X)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Coordinate X')
        ax1.set_title('Displacement along the X direction')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(time, VX)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Velocity VX')
        ax2.set_title('Velocity along the X direction')

        self.figures.append(fig1)
        self.figures.append(fig2)

    def show(self):
        for fig in self.figures:
            fig.show()
        plt.show()


class particle:
    def __init__(self, N, Nt, time_step=0):
        self.position = np.empty((Nt+1, 2))
        self.velocity = np.empty((Nt+1, 2))
        init_pos = [np.random.uniform(0, L * 1e-2), np.random.uniform(0, a * 1e-2)]
        self.position[:time_step+1] = init_pos
        self.velocity[:time_step+1] = [0.001, 0.001]
        self.out = False

    def update_potential(self, X, Y, epsilon, time_step):
        r = np.sqrt((self.position[time_step][0] - X) ** 2 + (self.position[time_step][1] - Y) ** 2)
        if not self.out:
            r = np.sqrt((self.position[time_step][0] - X)**2 + (self.position[time_step][1] - Y)**2)
            r[r <1e-10] = 1e-10
            potential = self.charge / (4 * np.pi * epsilon * r)
            return potential
        else:
            return np.zeros_like(r)

class Cation(particle):
    def __init__(self, N, Nt, time_step=0):
        super().__init__(N, Nt, time_step=time_step)
        self.charge = 1.602e-19
        self.mass = 3.8175458e-26

class Anion(particle):
    def __init__(self, N, Nt, time_step=0):
        super().__init__(N, Nt, time_step=time_step)
        self.charge = -1.602e-19
        self.mass = 5.8871086e-26


test1 = Data(a, L, t_obs, N, Nt, V0)

test1.run(1, 0, 10)
test1.show()

