import torch
import numpy as np
import math
from functools import reduce

eps = 10e-4
eps2 = 10e-6
KL_SCALER = 10.0
MIN_POINTS = 20
RADIUS = 0.5
NREGIONS = 3


def region_mean(num_regions):
    """
    Input:
        num_regions - number of regions
    Return:
        means of regions 
    """
    
    n = num_regions
    lookup = []
    d = 2 / n  # the cube size length
    #  construct all possibilities on the line [-1, 1] in the 3 axes
    for i in range(n-1, -1, -1):
        for j in range(n-1, -1, -1):
            for k in range(n-1, -1, -1):
                lookup.append([1 - d * (i + 0.5), 1 - d * (j + 0.5), 1 - d * (k + 0.5)])
    lookup = np.array(lookup)
    return lookup


def assign_region_to_point(X, device):
    """
    Input:
        X: point cloud [B, C, N]
        device: cuda:0, cpu
    Return:
        Y: Region assignment per point
    """

    n = NREGIONS
    d = 2 / n
    X_clip = torch.clamp(X, -0.99999999, 0.99999999)
    batch_size, _, num_points = X.shape
    Y = torch.zeros((batch_size, num_points), device=device, dtype=torch.long)  # label matrix

    # The code below partitions all points in the shape to voxels.
    # At each iteration find per axis the lower threshold and the upper threshold values
    # of the range according to n (e.g., if n=3, then: -1, -1/3, 1/3, 1 - there are 3 ranges)
    # and save points in the corresponding voxel if they fall in the examined range for all axis.
    region_id = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0, :]
                x_axis_ut = X_clip[:, 0, :] < -1 + (x + 1) * d
                y_axis_lt = -1 + y * d < X_clip[:, 1, :]
                y_axis_ut = X_clip[:, 1, :] < -1 + (y + 1) * d
                z_axis_lt = -1 + z * d < X_clip[:, 2, :]
                z_axis_ut = X_clip[:, 2, :] < -1 + (z + 1) * d
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = torch.cat([x_axis_lt, x_axis_ut, y_axis_lt, y_axis_ut,
                                z_axis_lt, z_axis_ut], dim=1).view(batch_size, 6, -1)
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask, _ = torch.min(in_range, dim=1)
                Y[mask] = region_id  # label each point with the region id
                region_id += 1

    return Y


def collapse_to_point(x, device):
    """
    Input:
        X: point cloud [C, N]
        device: cuda:0, cpu
    Return:
        x: A deformed point cloud. Randomly sample a point and cluster all point
        within a radius of RADIUS around it with some Gaussian noise.
        indices: the points that were clustered around x
    """
    # get pairwise distances
    inner = -2 * torch.matmul(x.transpose(1, 0), x)
    xx = torch.sum(x ** 2, dim=0, keepdim=True)
    pairwise_distance = xx + inner + xx.transpose(1, 0)

    # get mask of points in threshold
    mask = pairwise_distance.clone()
    mask[mask > RADIUS ** 2] = 100
    mask[mask <= RADIUS ** 2] = 1
    mask[mask == 100] = 0

    # Choose only from points that have more than MIN_POINTS within a RADIUS of them
    pts_pass = torch.sum(mask, dim=1)
    pts_pass[pts_pass < MIN_POINTS] = 0
    pts_pass[pts_pass >= MIN_POINTS] = 1
    indices = (pts_pass != 0).nonzero()

    # pick a point from the ones that passed the threshold
    point_ind = np.random.choice(indices.squeeze().cpu().numpy())
    point = x[:, point_ind]  # get point
    point_mask = mask[point_ind, :]  # get point mask

    # draw a gaussian centered at the point for points falling in the region
    indices = (point_mask != 0).nonzero().squeeze()
    x[:, indices] = torch.tensor(draw_from_gaussian(point.cpu().numpy(), len(indices)), dtype=torch.float).to(device)
    return x, indices


def draw_from_gaussian(mean, num_points):
    """
    Input:
        mean: a numpy vector
        num_points: number of points to sample
    Return:
        points sampled around the mean with small std
    """
    return np.random.multivariate_normal(mean, np.eye(3) * 0.001, num_points).T


def draw_from_uniform(gap, region_mean, num_points):
    """
    Input:
        gap: a numpy vector of region x,y,z length in each direction from the mean
        region_mean:
        num_points: number of points to sample
    Return:
        points sampled uniformly in the region
    """
    return np.random.uniform(region_mean - gap, region_mean + gap, (num_points, 3)).T


def farthest_point_sample(args, xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = torch.device("cuda:" + str(xyz.get_device()) if args.cuda else "cpu")

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals


def farthest_point_sample_np(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        dist = np.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=1)  # get the index of the point farthest away
    return centroids, centroids_vals


def rotate_shape(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')


def random_rotate_one_axis(X, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype('float32')


def translate_pointcloud(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
    Return:
        A translated shape
    """
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud.astype('float32')


def scale_to_unit_cube(x):
    """
   Input:
       x: pointcloud data, [B, C, N]
   Return:
       A point cloud scaled to unit cube
   """
    if len(x) == 0:
        return x

    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x


def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)

def rotate_point_cloud_by_angle_xyz(batch_data, angle_x=0, angle_y=0, angle_z=0):
    """ Rotate the point cloud along up direction with certain angle.
        Rotate in the order of x, y and then z.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = batch_data.reshape((-1, 3))
    
    cosval = np.cos(angle_x)
    sinval = np.sin(angle_x)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    rotated_data = np.dot(rotated_data, rotation_matrix)

    cosval = np.cos(angle_y)
    sinval = np.sin(angle_y)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(rotated_data, rotation_matrix)

    cosval = np.cos(angle_z)
    sinval = np.sin(angle_z)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(rotated_data, rotation_matrix)
    
    return rotated_data.reshape(batch_data.shape)


def rotate_point_cloud_by_angle_list(batch_data, rotation_angles):
    """ Rotate the point cloud along up direction with certain angle list.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angles[k])
        sinval = np.sin(rotation_angles[k])
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_axis_angle(batch_data_, u_, theta_):
    """ Rotate the point cloud around the axis u by angle theta
        u is a list of tuples, B * (x,y,z), consisting of unit vectors of the axis
        theta is a list of angles, B length array, representing the angle
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    batch_data = batch_data_.detach().numpy()
    u = u_
    theta = theta_
    u = np.squeeze(u)
    theta = np.squeeze(theta)
    eps = 1e-6
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        x, y, z = u[k]
        assert 1-eps < x*x+y*y+z*z < 1+eps

        cosval = np.cos(theta[k])
        sinval = np.sin(theta[k])

        rotation_matrix = np.array([[cosval + x*x*(1-cosval), x*y*(1-cosval) - z*sinval, x*z*(1-cosval) + y*sinval],
                                    [y*x*(1-cosval) + z*sinval, cosval + y*y*(1-cosval), y*z*(1-cosval) - x*sinval],
                                    [z*x*(1-cosval) - y*sinval, z*y*(1-cosval) + x*sinval, cosval + z*z*(1-cosval)]])
        

        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    
    return torch.tensor(rotated_data, dtype=batch_data_.dtype)
