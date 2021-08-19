import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import RotationAroundAxis
## https://stackoverflow.com/questions/14926798/python-transformation-matrix

def signed_power_function(x,toThePower):
    return np.sign(x) * (np.abs(x)** toThePower)


def octant_posNeg(xs,ys,zs):
    a = 1
    x_pos,y_pos,z_pos = [],[],[]
    x_neg,y_neg,z_neg=[],[],[]
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            if(xs[i][j]<0):
                a *= -1
            if(ys[i][j]<0):
                a *= -1
            if(zs[i][j]<0):
                a *= -1
            if a >0:
                x_pos.append(xs[i][j])
                y_pos.append(ys[i][j])
                z_pos.append(zs[i][j])
            else:
                x_neg.append(xs[i][j])
                y_neg.append(ys[i][j])
                z_neg.append(zs[i][j])
    pos_shape = int(np.floor(np.sqrt(len(x_pos))))
    neg_shape = int(np.floor(np.sqrt(len(x_neg))))
    print("len(x_pos): ", len(x_pos))
    print("len(x_neg): ", len(x_neg))
    print("len(y_pos): ", len(y_pos))
    print("len(y_neg): ", len(y_neg))
    print("len(z_pos): ", len(z_pos))
    print("len(z_neg): ", len(z_neg))

    print("pos_shape: ",pos_shape)
    print("neg_shape: ",neg_shape)
    end_idx_pos = int(pos_shape * pos_shape)
    end_idx_neg = int(neg_shape * neg_shape)
    x_pos,y_pos,z_pos = np.array(x_pos)[:end_idx_pos].reshape(pos_shape,pos_shape),np.array(y_pos)[:end_idx_pos].reshape(pos_shape,pos_shape),np.array(z_pos)[:end_idx_pos].reshape(pos_shape,pos_shape)
    x_neg,y_neg,z_neg = np.array(x_neg)[:end_idx_neg].reshape(neg_shape,neg_shape),np.array(y_neg)[:end_idx_neg].reshape(neg_shape,neg_shape),np.array(z_neg)[:end_idx_neg].reshape(neg_shape,neg_shape)

    return x_pos,y_pos,z_pos,x_neg,y_neg,z_neg

sizes = np.array([0.6,0.8,0.4])
shape = [0.9,0.5]

theta,phi = np.mgrid[-np.pi/2:np.pi/2:80j, -np.pi: np.pi :240j ] ## theta = [-pi/2, pi/2]; phi=[-pi, pi]
x = sizes[0] * signed_power_function(np.cos(theta),shape[0]) * signed_power_function(np.cos(phi),shape[1])
y = sizes[1] * signed_power_function(np.cos(theta),shape[0]) * signed_power_function(np.sin(phi),shape[1])
z = sizes[2] * signed_power_function(np.sin(theta), shape[0])
print("x.shape: ",x.shape)
print("y.shape: ",y.shape)
x_pos,y_pos,z_pos,x_neg,y_neg,z_neg = octant_posNeg(x,y,z)
# print("x_pos.shape: ",x_pos.shape)
# print("x_neg.shape: ",x_neg.shape)

x_rot,y_rot,z_rot = RotationAroundAxis.get_rotated_sq(x,y,z,10)
x_rot_pos,y_rot_pos,z_rot_pos, x_rot_neg, y_rot_neg, z_rot_neg = octant_posNeg(x_rot,y_rot,z_rot)


fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.plot_surface(x_pos,y_pos,z_pos,color="blue")
ax.plot_surface(x_neg,y_neg,z_neg,color="red")
ax.plot_surface(x_rot_pos,y_rot_pos,z_rot_pos,color="blue")
ax.plot_surface(x_rot_neg,y_rot_neg,z_rot_neg,color="red")
## Adjustment of the axes, so they all have the same span
max_radius = np.max(sizes)
for axis in 'xyz':
    getattr(ax,'set_{}lim'.format(axis))(-max_radius, max_radius)
plt.show()



### Combination of positive and negatives are not working for shape mismatch ###
print(f"x_neg.shape:{x_neg.shape},y_pos.shape: {y_pos.shape},z_neg.shape: {z_neg.shape}")
##      x_neg.shape:(69, 69),y_pos.shape: (120, 120),z_neg.shape: (69, 69)

fig = plt.figure(figsize=(10,12))
nrows,ncols = 2,2
ax1 = fig.add_subplot(nrows,ncols,1,projection="3d")
ax1.plot_surface(x_pos,y_pos,z_pos)
ax1.plot_surface(x_neg,y_neg,z_neg)
# ax1.plot_surface(x_pos,y_neg,z_pos)
# ax1.plot_surface(x_neg,y_pos,z_neg)
ax1.set_title("x_pos,y_pos,z_pos and x_neg,y_neg,z_neg")

ax2 = fig.add_subplot(nrows,ncols,2,projection="3d")
ax2.plot_surface(x_pos,y_neg,z_pos)
ax2.plot_surface(x_neg,y_pos,z_neg)
ax2.set_title("x_pos,y_neg,z_pos and x_neg,y_pos,z_neg")

ax3 = fig.add_subplot(nrows,ncols,3,projection="3d")
ax3.plot_surface(x_pos,y_neg,z_pos)
ax3.plot_surface(x_neg,y_neg,z_neg)
ax3.set_title("x_pos,y_neg,z_pos and x_neg,y_neg,z_neg")

plt.show()