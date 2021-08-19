import numpy as np

def rotate_xAxis(theta):
    rotation_mat = np.array([[1,        0,             0      ],
                             [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta), np.cos(theta)]
                             ])
    return rotation_mat

def rotate_yAxis(theta):
    rotation_mat = np.array([ [np.cos(theta), 0, np.cos(theta)],
                              [0            , 1,        0     ],
                              [-np.sin(theta),0, np.cos(theta)]
                              ])
    return rotation_mat

def rotate_zAxis(theta):
    rotation_mat = np.array([ [np.cos(theta), -np.sin(theta),   0],
                              [np.sin(theta),  np.cos(theta),   0],
                              [     0       ,        0      ,   1]
                              ])
    return rotation_mat

def get_rotated_sq(x_coo,y_coo,z_coo,theta_rot,rot_around_axis="z"):
    if(rot_around_axis=="z"):
        rot_axis = rotate_xAxis(theta_rot)
    elif(rot_around_axis=="y"):
        rot_axis = rotate_yAxis(theta_rot)
    elif(rot_around_axis=="z"):
        rot_axis =rotate_zAxis(theta_rot)
    else:
        print("!!! Please Specify the Axis along which you want to rotate !!!")
    rotated_mat = np.zeros((3,x_coo.shape[0] * x_coo.shape[1]))
    col_idx = 0
    for i in range(x_coo.shape[0]):
        for j in range(x_coo.shape[1]):
            coo_vec = np.array([ [x_coo[i][j]], [y_coo[i][j]], [z_coo[i][j]] ]).squeeze()
            rotated_vec = np.dot(rot_axis,coo_vec)
            rotated_mat[:,col_idx] = rotated_vec
            col_idx+=1
    x_rot = np.reshape(rotated_mat[0],(x_coo.shape[0], x_coo.shape[1]))
    y_rot = np.reshape(rotated_mat[1],(x_coo.shape[0], x_coo.shape[1]))
    z_rot = np.reshape(rotated_mat[2],(x_coo.shape[0], x_coo.shape[1]))

    return x_rot, y_rot, z_rot