from zinnia import *
import numpy as np

@zk_chip
def getAcc(pos: Float[Array, 'N<=1024 C<=3'], mass: Float[Array, 'N<=1024'], G: Float, softening: Float):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    inv_r3 = dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass
    a = np.hstack((ax, ay, az))
    return a

@zk_chip
def getEnergy(pos: Float[Array, 'N<=1024 C<=3'], vel: Float[Array, 'N<=1024 C<=3'], mass: Float[Array, 'N<=1024'], G: Float):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    KE = 0.5 * np.sum(mass * vel ** 2)
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    inv_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]
    PE = G * np.sum(np.triu(-(mass * mass.T) * inv_r, 1))
    return (KE, PE)

@zk_circuit
@requires(lambda mass, pos, vel, N, Nt, dt, G, softening: 0 <= N <= 1024)
@requires(lambda mass, pos, vel, N, Nt, dt, G, softening: dt != 0.0)
def nbody(mass: Float[Array, 'N<=1024'], pos: Float[Array, 'N<=1024 C<=3'], vel: Float[Array, 'N<=1024 C<=3'], N: Integer, Nt: Integer, dt: Float, G: Float, softening: Float):
    vel -= np.mean(mass * vel, axis=0) / np.mean(mass)
    acc = getAcc(pos, mass, G, softening)
    KE = np.ndarray(Nt + 1, dtype=np.float64)
    PE = np.ndarray(Nt + 1, dtype=np.float64)
    KE[0], PE[0] = getEnergy(pos, vel, mass, G)
    t = 0.0
    for i in range(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc = getAcc(pos, mass, G, softening)
        vel += acc * dt / 2.0
        t += dt
        KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)
    return (KE, PE)
