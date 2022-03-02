#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np
from scipy import interpolate


def integrateFp(pressure, gridx, gridy, center, R, N_grid=1000):
    """
    Integrate - p * n_vec dS
    """
    # print(np.shape(pressure))
    # print("Check symmetry ...")
    # pressure_up = pressure[:256]
    # pressure_down = pressure[256:]
    # pressure_down = pressure_down[::-1]
    # print(np.linalg.norm(pressure_down - pressure_up))
    perimeter = 2 * np.pi * R
    dtheta = 2 * np.pi / N_grid
    ds = perimeter / N_grid
    dx = gridx[1] - gridx[0]
    dy = gridy[1] - gridy[0]
    integral_values = []
    theta_values = []
    center_x, center_y = center
    integral = 0.0
    theta = 0.0
    meshx, meshy = np.meshgrid(gridx, gridy)
    # plt.contourf(meshx, meshy, pressure)
    # plt.plot(center_x, center_y, "rx", markersize=10, linewidth=8)
    for ni in range(N_grid):
        nx = np.cos(theta)
        ny = np.sin(theta)
        x = center_x + nx * R
        y = center_y + ny * R
        i = np.argmin(np.abs(x - gridx))
        j = np.argmin(np.abs(y - gridy))
        # Building matrix for interpolation
        Xind = [i - 3, i - 2, i - 1, i, i + 1, i + 2, i + 3]
        Yind = [j - 3, j - 2, j - 1, j, j + 1, j + 2, j + 3]
        XXind, YYind = np.meshgrid(Xind, Yind)
        PP = pressure[YYind, XXind]
        X = gridx[Xind]
        Y = gridy[Yind]
        f = interpolate.interp2d(X, Y, PP, kind='linear')
        # f = interpolate.interp2d(X, Y, PP, kind='cubic')
        p = f(x, y)[0]
        n_vec = np.array([nx, ny])
        n_vec = np.reshape(n_vec, (2))
        # integral_value = (- p * n_vec * ds)
        # integral_value = n_vec * ds
        integral_value = (-p * n_vec * ds)
        integral += integral_value
        theta += dtheta
        integral_values.append(integral_value)
        theta_values.append(theta)
        # plt.plot(x, y, "rx", markersize=3, linewidth=8)
        # plt.plot(center_x + nx, center_y + ny, "bo", markersize=3, linewidth=8)
        # print("Interpolated pressure {:.2f}".format(p))
        # print("n_vec {:}".format(n_vec))
    # print(integral)
    # plt.colorbar()
    # plt.show()
    # plt.plot(theta_values, np.array(integral_values)[:,0]*0.15)
    # plt.title("$F_p$")
    # plt.show()
    return integral


def getLiftedSurfaceValues(array, grid, center, Rmin, Rmax):
    center_x, center_y = center
    num_gridpoints_in_circle = 0
    Nx, Ny = np.shape(array)
    x1 = []
    y1 = []
    values = []
    for i in range(Nx):
        for j in range(Ny):
            xi = grid[0][i]
            yi = grid[1][j]
            # Check if point is inside circle
            radius_ = np.sqrt(
                np.power(xi - center_x, 2.0) + np.power(yi - center_y, 2.0))
            if radius_ <= Rmin:
                num_gridpoints_in_circle += 1
                # print("found point inside circle")
            elif radius_ <= Rmax:
                x1.append(xi)
                y1.append(yi)
                value = array[i, j]
                values.append(value)
            else:
                pass
    # print("Found {:}/{:} gridpoints in circle. {:.2f}%".format(num_gridpoints_in_circle, Nx * Ny, num_gridpoints_in_circle/Nx/Ny*100.))
    # print(np.shape(x1))
    # print(np.shape(y1))
    # print(np.shape(values))
    points = np.array((x1, y1)).T
    return points, values


def checkSymmetry(data):
    data_up = data[:np.shape(data)[0] // 2]
    data_down = data[np.shape(data)[0] // 2:]
    data_down = data_down[::-1]
    print(np.linalg.norm(data_down - data_up))
    return 0


def integrateFm(velocity, gridx, gridy, center, R, mu, N_grid=360 * 4):
    ########################################################
    ## Integrate mu (\Grad u + \Gradu^T) * n_vec dS
    ########################################################
    # print(np.shape(velocity))
    # print("# integrateFm() #")
    vx = velocity[0]
    vy = velocity[1]
    perimeter = 2 * np.pi * R
    dtheta = 2 * np.pi / N_grid
    ds = perimeter / N_grid
    dx = gridx[1] - gridx[0]
    dy = gridy[1] - gridy[0]
    center_x, center_y = center

    theta = 0.0
    grid_perimeter = []
    # Get points around the circle to interpolate
    for ni in range(N_grid):
        nx = np.cos(theta)
        ny = np.sin(theta)
        x = center_x + nx * R
        y = center_y + ny * R
        theta += dtheta
        grid_perimeter.append([x, y])

    # plt.imshow(vx)
    # plt.show()
    #
    # print(np.shape(vx))
    # print("Check symmetry ...")
    # vx_up = vx[:256]
    # vx_down = vx[256:]
    # vx_down = vx_down[::-1]
    # print(np.linalg.norm(vx_down - vx_up))

    # # plt.imshow(vy)
    # # plt.show()

    # vy_up = vy[:256]
    # vy_down = vy[256:]
    # vy_down = vy_down[::-1]
    # print(np.linalg.norm(vy_down - vy_up))

    # print(ark)

    vx = vx.T
    vy = vy.T
    """ Computing the gradient of the velocity in x direction """

    # Grid
    gridvx = gridx[1:-1]
    gridvy = gridy[1:-1]

    vx_gradx = (vx[2:] - vx[:-2]) / (2 * dx)
    vx_gradx_grid = [gridvx, gridy]

    vx_grady = (vx[:, 2:] - vx[:, :-2]) / (2 * dy)
    vx_grady_grid = [gridx, gridvy]

    vy_gradx = (vy[2:] - vy[:-2]) / (2 * dx)
    vy_gradx_grid = [gridvx, gridy]

    vy_grady = (vy[:, 2:] - vy[:, :-2]) / (2 * dy)
    vy_grady_grid = [gridx, gridvy]

    # print(np.shape(vx_gradx))
    # print(np.shape(vx_gradx_grid[0]), np.shape(vx_gradx_grid[1]))

    # print(np.shape(vx_grady))
    # print(np.shape(vx_grady_grid[0]), np.shape(vx_grady_grid[1]))

    # print(np.shape(vy_gradx))
    # print(np.shape(vy_gradx_grid[0]), np.shape(vy_gradx_grid[1]))

    # print(np.shape(vy_grady))
    # print(np.shape(vy_grady_grid[0]), np.shape(vy_grady_grid[1]))

    # checkSymmetry(vx_gradx.T)
    # checkSymmetry(vx_grady.T)
    # checkSymmetry(vy_gradx.T)
    # checkSymmetry(vy_grady.T)

    # plt.imshow(vx_gradx.T)
    # plt.show()

    # plt.imshow(vx_grady.T)
    # plt.show()

    # plt.imshow(vy_gradx.T)
    # plt.show()

    # plt.imshow(vy_grady.T)
    # plt.show()

    # print(ark)
    """
    Computing the gradients of u in a lifted surface.
    Points inside the lifter surface, are ignored in the interpolation.
    """
    R_lifted = R + 1 * dx
    R_lifted_max = R + 3 * dx
    kind = 'nearest'  # 'nearest' 'linear' 'cubic' 'quintic'

    # tempx, tempy = np.meshgrid(vx_gradx_grid[0], vx_gradx_grid[1])
    # points = np.array((tempx.T.flatten(), tempy.T.flatten())).T
    # values = vx_gradx.flatten()
    # vx_gradx_perimeter = interpolate.griddata(points, values, grid_perimeter, method=kind)

    points, values = getLiftedSurfaceValues(vx_gradx, vx_gradx_grid, center,
                                            R_lifted, R_lifted_max)
    vx_gradx_perimeter = interpolate.griddata(points,
                                              values,
                                              grid_perimeter,
                                              method=kind)

    points, values = getLiftedSurfaceValues(vx_grady, vx_grady_grid, center,
                                            R_lifted, R_lifted_max)
    vx_grady_perimeter = interpolate.griddata(points,
                                              values,
                                              grid_perimeter,
                                              method=kind)

    points, values = getLiftedSurfaceValues(vy_gradx, vy_gradx_grid, center,
                                            R_lifted, R_lifted_max)
    vy_gradx_perimeter = interpolate.griddata(points,
                                              values,
                                              grid_perimeter,
                                              method=kind)

    points, values = getLiftedSurfaceValues(vy_grady, vy_grady_grid, center,
                                            R_lifted, R_lifted_max)
    vy_grady_perimeter = interpolate.griddata(points,
                                              values,
                                              grid_perimeter,
                                              method=kind)

    # print(np.shape(vx_gradx_perimeter))
    # print(np.shape(vx_grady_perimeter))
    # print(np.shape(vy_gradx_perimeter))
    # print(np.shape(vy_grady_perimeter))
    # print(np.shape(grid_perimeter))

    integral = 0.0
    theta = 0.0
    for ni in range(N_grid):
        # print("-"*10)
        nx = np.cos(theta)
        ny = np.sin(theta)

        n_vec = np.array([nx, ny])
        n_vec = np.reshape(n_vec, (2))

        vxdx = vx_gradx_perimeter[ni]
        vxdy = vx_grady_perimeter[ni]
        vydx = vy_gradx_perimeter[ni]
        vydy = vy_grady_perimeter[ni]
        nablav = np.array([[vxdx, vydx], [vxdy, vydy]])

        nablav = (nablav + nablav.T)
        integral_value = np.dot(nablav, n_vec) * ds

        # px, py = grid_perimeter[ni]
        # print(px - (center_x + nx * R))
        # print(py - (center_y + ny * R))
        # print(grid_perimeter[ni])
        # print(vxdx)

        # integral_value = ds
        # integral_value = vxdx * ny * ds

        integral += integral_value
        theta += dtheta

    integral *= mu
    return integral
