"""

    PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl 
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from .smal_basics import align_smal_template_to_symmetry_axis #, get_smal_template
from global_utils import config

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

class SMAL(nn.Module):
    def __init__(self, device, shape_family_id=-1, dtype=torch.float):
        super(SMAL, self).__init__()

        # -- Load SMPL params --
        # with open(pkl_path, 'r') as f:
        #     dd = pkl.load(f)

        print (f"Loading SMAL with shape family: {shape_family_id}")
            
        with open(config.SMAL_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            dd = u.load()

        self.f = dd['f']

        self.faces = torch.from_numpy(self.f.astype(int)).to(device)
        
        # replaced logic in here (which requried SMPL library with L58-L68)
        # v_template = get_smal_template(
        #     model_name=config.SMAL_FILE, 
        #     data_name=config.SMAL_DATA_FILE, 
        #     shape_family_id=shape_family_id)

        v_template = dd['v_template']

        # Size of mesh [Number of vertices, 3]
        self.size = [v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis
        
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T.copy()
        self.shapedirs = Variable(
            torch.Tensor(shapedir), requires_grad=False).to(device)

        with open(config.SMAL_DATA_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()

        # Zero_Betas -> V_Template -> Aligned
        # Zero_Betas -> V_Template -> V_Template + ShapeCluster * ShapeDirs -> Aligned

        # Aligned(V_T + ShapeCluster * ShapeDirs) - ShapeCluster * ShapeDirs

        # Select mean shape for quadruped type
        shape_cluster_means = data['cluster_means'][shape_family_id]
        
        # NOTE: The model was trained using v_template for shape_params=[0]*41
        if shape_family_id != -1:
            v_template = v_template + np.matmul(
                shape_cluster_means[None,:], shapedir).reshape(
                -1, self.size[0], self.size[1])[0]

        self.shape_cluster_means = torch.from_numpy(shape_cluster_means).float().to(device)

        v_sym, self.left_inds, self.right_inds, self.center_inds = align_smal_template_to_symmetry_axis(
            v_template, sym_file=config.SMAL_SYM_FILE)

        # Mean template vertices
        self.v_template = Variable(
            torch.Tensor(v_sym),
            requires_grad=False).to(device)

        # Regressor for joint locations given shape 
        self.J_regressor = Variable(
            torch.Tensor(dd['J_regressor'].T.todense()),
            requires_grad=False).to(device)

        # Pose blend shape basis
        num_pose_basis = dd['posedirs'].shape[-1]
        
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = Variable(
            torch.Tensor(posedirs), requires_grad=False).to(device)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = Variable(
            torch.Tensor(undo_chumpy(dd['weights'])),
            requires_grad=False).to(device)

        self.first_call = True
        self.fig = None
        self.ax = None
        self.colors = np.array([[1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[0.0, 1.0, 0.0, 1.0],[0.0, 1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[0.0, 0.0, 1.0, 1.0],[0.0, 1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[0.0, 0.0, 1.0, 1.0],[1.0, 0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 1.0],[0.0, 1.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0, 1.0],[0.0, 1.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0, 1.0],[0.0, 1.0, 0.0, 1.0],[1.0, 1.0, 0.0, 1.0],[1.0, 1.0, 0.0, 1.0],
                                [1.0, 1.0, 0.0, 1.0],[1.0, 1.0, 0.0, 1.0],[1.0, 1.0, 0.0, 1.0],[1.0, 1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0]])

        self.count = 0


    def __call__(self, beta, theta, trans=None, del_v=None, betas_logscale=None, get_skin=True, v_template=None):

        if True:
            nBetas = beta.shape[1]
        else:
            nBetas = 0

        
        # v_template = self.v_template.unsqueeze(0).expand(beta.shape[0], 3889, 3)
        if v_template is None:
            v_template = self.v_template

        # 1. Add shape blend shapes
        
        if nBetas > 0:
            if del_v is None:
                v_shaped = v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
            else:
                v_shaped = v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas,:]), [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = v_template.unsqueeze(0)
            else:
                v_shaped = v_template + del_v 


        ## joints are obtained by regressing from the vertices of the mesh. Do we need to do that?
        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # figure out transform somehow

        #for j in J[0]:
        #    xyz = j.numpy()
        #    print(str(xyz[0]) + ",\t" + str(xyz[1]) + ",\t" + str(xyz[2]) + ",\t", end='')
        #print("")
        # print('after step two, J size: ', np.shape(J))  # size [1, 35, 3]
        # 3. Add pose blend shapes
        # N x 24 x 3 x 3


        ## Rs have to do with rotation?
        # Obtaining rotation matrices for each joint from the Rodrigues representation
        if len(theta.shape) == 4:
            Rs = theta
        else:
            Rs = torch.reshape(batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3])
        
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(beta.device), [-1, 306])
        
        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        # print('parents: ', self.parents)

        #4. Get the global joint location (only place parents is used)
        self.J_transformed, A = batch_global_rigid_transformation(
            Rs, J, self.parents, betas_logscale=betas_logscale)
        # A is the ordered set of joint ancestors
        '''
        if self.first_call:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.first_call = False
        new_joints = self.J_transformed[0]
        print('joints: ', new_joints)
        plt.cla()
        self.ax.scatter(new_joints[:, 0], new_joints[:, 2], -new_joints[:, 1], c=self.colors)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.view_init(0, 0)
        self.ax.set_xlim(-.75, 0.0)
        self.ax.set_ylim(-1, 0.5)
        self.ax.set_zlim(-.5, 0.5)
        plt.savefig('/home/masselmeier/Desktop/SP22/CS8803/outputs/' + str(self.count) + '.png')
        self.count += 1
        # print('J transformed shape: ', np.shape(self.J_transformed))  # size [1, 35, 3]
        # print('J_transformed: ', self.J_transformed)
        '''
        # 5. Do skinning:
        num_batch = theta.shape[0]
        
        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

            
        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])),
                [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
                [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=beta.device)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch,3)).to(device=beta.device)

        verts = verts + trans[:,None,:]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        '''
        output_txt = '/home/masselmeier/Desktop/SP22/CS8803/motion_imitation/retarget_motion/data/WLDO_output.txt'
        with open(output_txt, 'a') as f:
            for i in range(0, len(joints[0])):
                j = joints[0][i]
                xyz = j.numpy()
                f.write(str(xyz[0]) + ",\t" + str(xyz[1]) + ",\t" + str(xyz[2]))
                if i < len(joints[0]) - 1:
                    f.write(",\t")
            f.write('\n')
        '''
        # print('final joints: ', joints)  # size [1, 35, 3]
        joints = torch.cat([
            joints,
            verts[:, None, 1863], # end_of_nose
            verts[:, None, 26], # chin
            verts[:, None, 2124], # right ear tip
            verts[:, None, 150], # left ear tip
            verts[:, None, 3055], # left eye
            verts[:, None, 1097], # right eye
            ], dim = 1) 

        # print('final joints shape: ', np.shape(joints))
        if get_skin:
            return verts, joints, Rs, v_shaped
        else:
            return joints











