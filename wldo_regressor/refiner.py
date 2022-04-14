"""
Similar to RunModel, but fine-tunes over time on openpose output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from .ops import keypoint_l1_loss
from global_utils.refiner_models import Encoder_fc3_dropout
from global_utils.smal_model.batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from global_utils.smal_model.smal_torch import SMAL

from wldo_regressor.demo import load_model_from_disk

from dataset_dogs.demo_dataset import DemoDataset
from torch.utils.data import DataLoader
#from .tf_smpl.batch_lbs import batch_rodrigues
#from .tf_smpl.batch_smpl import SMPL
#from .tf_smpl.projection import batch_orth_proj_idrot
#from .util.renderer import SMPLRenderer, draw_skeleton
#from .util.image import unprocess_image
import time
from os.path import exists
from tqdm import tqdm

import tensorflow as tf
import numpy as np

import torch
import torch.nn as nn

class Refiner(object):
    def __init__(self, dataset, config, num_frames, args, device):
        """
        Args:
          config,,
        """
        '''
        # Config + path
        if not config.load_path:
            raise Exception(
                "[!] You should specify `load_path` to load a pretrained model")
        if not exists(config.load_path + '.index'):
            print('%s doesnt exist..' % config.load_path)
            import ipdb
            ipdb.set_trace()
        '''
        self.args = args
        self.config = config
        self.device = device
        # self.load_path = config.load_path
        self.num_frames = num_frames

        self.data_format = config.data_format
        # self.smpl_model_path = config.smpl_model_path

        # Loss & Loss weights:
        self.e_lr = config.e_lr

        # self.e_loss_weight = config.e_loss_weight
        self.shape_loss_weight = config.shape_loss_weight
        self.joint_smooth_weight = config.joint_smooth_weight
        self.camera_smooth_weight = config.camera_smooth_weight
        # self.keypoint_loss = keypoint_l1_loss
        self.init_pose_loss_weight = config.init_pose_loss_weight

        # Data
        self.batch_size = num_frames
        self.img_size = config.img_size
        '''
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  # , num_workers=num_workers)
        tqdm_iterator = tqdm(self.data_loader, desc='Eval', total=len(self.data_loader))
        # only iterates once
        for step, batch in enumerate(tqdm_iterator):
            self.batch = batch
        '''
        input_size = [self.batch_size, 3, self.img_size, self.img_size]  # [self.batch_size, self.img_size, self.img_size, 3]
        self.images = torch.zeros(input_size, dtype=torch.float32)  #tf.placeholder(tf.float32, shape=input_size)
        self.img_feat = torch.zeros([self.batch_size, 2048], dtype=torch.float32)

        # Camera type!
        self.num_cam = 3
        # self.proj_fn = batch_orth_proj_idrot
        self.num_theta = 105  # 24 * 3, 35 * 3
        self.num_beta = 26 # 10
        self.total_params = + self.num_cam + self.num_theta + self.num_beta  # (134) 85 in total

        self.smal = SMAL(self.device, shape_family_id=self.args.shape_family_id)
        self.WLDO_model = load_model_from_disk(self.args.checkpoint, self.args.shape_family_id, False, self.device)

        # hypothesis: theta0 is the initial pose outputs, theta_var is the refined one?
        self.theta0_shape = [self.batch_size, self.total_params]
        self.theta0 = self.load_mean_param()
        self.theta_prev = self.theta0  # size: 85, need to have this be actual theta0

        # Optimization space.
        self.refine_inpose = config.refine_inpose
        if self.refine_inpose:
            self.theta = torch.zeros(self.theta0_shape, dtype=torch.float32)   #tf.get_variable("theta_var", dtype=tf.float32, shape=self.theta0_pl_shape)

        # Setup optimizer
        print('Setting up optimizer..')

        # TODO: change this to latent space parameters?
        #print('WLDO model parameters: ')
        #for name, param in self.WLDO_model.module.netG_DETAIL.code_predictor.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)
        self.e_optimizer = torch.optim.Adam(self.WLDO_model.module.netG_DETAIL.code_predictor.parameters(), lr=self.e_lr)  # self.optimizer = tf.train.AdamOptimizer
        # e_optimizer = self.optimizer(self.e_lr)

        # For ft-loss
        # self.shape_pl = tf.placeholder_with_default(tf.zeros(10), shape=(10,), name='beta0')
        self.shape = torch.zeros([self.num_beta,], dtype=torch.float32)

        # For stick-to-init-pose loss:
        self.init_pose = torch.zeros([num_frames, self.num_theta], dtype=torch.float32)
        self.init_pose_weight = torch.zeros([num_frames, 1], dtype=torch.float32)

        # For camera loss
        self.scale_factors = torch.ones([num_frames], dtype=torch.float32)
        self.offsets = torch.zeros([num_frames, 2], dtype=torch.float32)

        # Build model!
        self.ief = config.ief  # just a boolean
        if self.ief:
            self.num_stage = config.num_stage
            # self.build_refine_model()

        # setting up for predict
        self.batch_size = 1
        self.theta_prev = self.load_mean_param()  # size: 85, need to have this be actual theta0



    def load_mean_param(self):
        # initializing scale=0.9, translation=0.0, thetas=0.0, betas=0.0
        # Assuming that motion sequence starts from all 0's?
        mean = np.zeros((1, self.total_params))
        mean[0, 0] = 0.9  # This is scale.
        init_mean = np.tile(mean, (self.batch_size, 1))
        init_mean = torch.FloatTensor(init_mean)

        return init_mean

    # RECONSTRUCTION LOSS CALCULATED HERE
    def build_refine_model(self):
        # LOADING IN 3D POSE ESTIMATOR
        #self.WLDO_model = #Encoder_resnet  # Original 3D pose predictor? "3D pose is predicted by first encoding an image I into a 2048D latent space .."
        #self.threed_enc_fn = Encoder_fc3_dropout(self.total_params*2, self.total_params)  # policy? not sure (my guess is, this is what changes the 3D poses)

        # Question: does self.E_var have anything to do with the initial state distribution?
        # aren't images_pl empty here?
        #self.img_feat, self.E_var = img_enc_fn(self.images_pl,is_training=False,reuse=False)
        # img_feat should be: camera, pose, shape

        # Start loop
        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.all_Jsmal = []
        self.final_thetas = []
        # why d
        theta_prev = self.theta0  # size: 3 + 105 + 26 = 134, are these values correct?
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            print('theta prev size: ', theta_prev.size())
            print('img feat size: ', self.img_feat.size())

            state = torch.cat((self.img_feat, theta_prev), 1)
            print('state data type: ', state.dtype)
            '''
            # I think the reuse variable here is tensorflow specific
            if i == 0:
                delta_theta, threeD_var = threed_enc_fn(state,num_output=self.total_params,is_training=False,reuse=False)
                self.E_var.append(threeD_var)
            else:
                delta_theta, _ = threed_enc_fn(state,num_output=self.total_params,is_training=False,reuse=True)
            '''
            delta_theta = self.threed_enc_fn(state)

            ## output of policy: delta_theta
            # Compute new theta
            theta_here = theta_prev + delta_theta

            # modifed values below
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]

            # shapes: betas_pred
            # poses: poses_pred
            # we may need trans_pred and betas_logscale here
            verts, Js, pred_Rs, _ = self.smal(shapes, poses) #, trans=trans_pred, betas_logscale=betas_logscale)
            Jsmal = self.smal.J_transformed

            # Project to 2D!
            self.all_verts.append(verts)
            self.all_cams.append(cams)
            self.all_Js.append(Js)
            self.all_Jsmal.append(Jsmal)
            # save each theta.
            self.final_thetas.append(theta_here)
            # Finally, update to end iteration.
            theta_prev = theta_here


        # Compute everything with the final theta.
        # not sure where theta_var is actually updated
        if self.refine_inpose:
            #self.set_theta_var = self.theta_var.assign(self.theta_pl)
            theta_final = self.theta_var
        else:
            theta_final = theta_here

        cams = theta_final[:, :self.num_cam]
        poses = theta_final[:, self.num_cam:(self.num_cam + self.num_theta)]
        shapes = theta_final[:, (self.num_cam + self.num_theta):]
        ## we may need trans_pred and betas_logscale

        verts, Js, pred_Rs, _ = self.smal(shapes, poses) #,trans=trans_pred,betas_logscale=betas_logscale)

        Jsmal = self.smal.J_transformed
        # Project to 2D!

        self.all_verts.append(verts)
        self.all_cams.append(cams)
        self.all_Js.append(Js)
        self.all_Jsmal.append(Jsmal)
        # save each theta.
        self.final_thetas.append(theta_final)

        # Beta variance should be low! (WHAT IS THIS)
        self.loss_shape = self.shape_loss_weight * shape_variance(shapes, self.shape)

        # (3D CONSISTENCY LOSS)
        self.loss_init_pose = self.init_pose_loss_weight * init_pose(pred_Rs, self.init_pose)
        # Endpoints should be smooth!! (SMOOTHNESS LOSS)
        self.loss_joints = self.joint_smooth_weight * joint_smoothness(Js)

        # Camera should be smooth (CAMERA SMOOTHNESS LOSS)
        self.loss_camera = self.camera_smooth_weight * camera_smoothness(cams, self.scale_factors, self.offsets,
                                                                         img_size=self.config.img_size)

        self.total_loss = self.loss_shape + self.loss_joints + self.loss_init_pose + self.loss_camera

        # Setup optimizer
        print('Setting up optimizer..')
        e_optimizer = torch.optim.Adam(self.threed_enc_fn.parameters(), lr=self.e_lr) #self.optimizer = tf.train.AdamOptimizer
        # e_optimizer = self.optimizer(self.e_lr)

        e_optimizer.zero_grad()
        self.total_loss.backward()

        '''
        # optimizing wrt theta?
        if self.refine_inpose:
            self.e_opt = e_optimizer.minimize(self.total_loss, var_list=[self.theta_var])
        else:
            self.e_opt = e_optimizer.minimize(self.total_loss, var_list=[self.img_feat_var])
        '''

        e_optimizer.step()
        print('Done initializing the model!')

    def train(self, batch):
        preds = self.WLDO_model(batch)
        poses = preds['pose']
        shapes = preds['betas']
        camera_pred = preds['camera']
        trans_pred = preds['trans']
        scale_pred = torch.unsqueeze(camera_pred[:, 0], 1)
        trans_x_pred = torch.unsqueeze(trans_pred[:, 0], 1)
        trans_y_pred = torch.unsqueeze(trans_pred[:, 1], 1)
        cams = torch.cat((scale_pred, trans_x_pred, trans_y_pred), 1)

        verts, Js, pred_Rs, _ = self.smal(shapes, poses)  # TO ADD: trans=trans_pred,betas_logscale=betas_logscale)

        # shape variance loss
        mean_shape = torch.mean(shapes, dim=0)
        self.loss_shape = self.shape_loss_weight * shape_variance(shapes, mean_shape)

        # (3D CONSISTENCY LOSS)
        # where is init_pose coming?
        self.loss_init_pose = self.init_pose_loss_weight * init_pose(pred_Rs, self.init_pose)

        # Endpoints should be smooth!! (SMOOTHNESS LOSS)
        self.loss_joints = self.joint_smooth_weight * joint_smoothness(Js)

        # Camera should be smooth (CAMERA SMOOTHNESS LOSS)
        self.loss_camera = self.camera_smooth_weight * camera_smoothness(cams, self.scale_factors, self.offsets,
                                                                         img_size=self.config.img_size)

        self.total_loss = self.loss_shape + self.loss_joints + self.loss_camera # + self.loss_init_pose

        self.e_optimizer.zero_grad()
        self.total_loss.backward()
        self.e_optimizer.step()

        return self.total_loss

    # predict on one image
    def predict(self, batch):
        """
        images: num_batch, img_size, img_size, 3
        kps: num_batch x 19 x 3
        Preprocessed to range [-1, 1]

        scale_factors, offsets: used to preprocess the bbox

        Runs the model with images.
        """
        # print('in predict: ')
        preds = self.WLDO_model(batch)
        # print('after model call ')
        poses = preds['pose']
        shapes = preds['betas']

        verts, Js, pred_Rs, _ = self.smal(shapes, poses)  # TO ADD: trans=trans_pred,betas_logscale=betas_logscale)
        # print('after smal call')

        Jsmal = self.smal.J_transformed
        # print('returning')
        return Jsmal

# All the  loss functions.

def shape_variance(shapes, target_shape=None):
    # Shapes is F x (26) 10
    # Compute variance.
    if target_shape is not None:
        #N = tf.shape(shapes)[0]
        #target_shapes = tf.tile(tf.expand_dims(target_shape, 0), [N, 1])
        N = shapes.size()[0]
        target_shapes = torch.unsqueeze(target_shape, 0)
        #target_shapes = torch.tile(, (N, 1))
        target_shapes = target_shapes.repeat(N, 1)
        print('target shapes size: ', target_shapes.size())
        print('shapes size: ', shapes.size())
        loss = nn.MSELoss(reduce=None)
        return loss(target_shapes, shapes)   #reduction='sum')  # tf.losses.mean_squared_error(target_shapes, shapes)
    else:
        #_, var = tf.nn.moments(shapes, axes=0)
        var = torch.var(shapes, dim=0)
        return torch.mean(var)  # tf.reduce_mean(var)


def joint_smoothness(joints):
    """
    joints: N x 35(?) for SMAL x 3
    Computes smoothness of joints relative to root.
    """
    # 7,11,17,21: four hips
    if joints.shape[1] == 35:
        upper_left_hip = 7
        upper_right_hip = 11
        rear_left_hip = 17
        rear_right_hip = 21
        # left_hip, right_hip = 3, 2
        #root = (joints[:, left_hip] + joints[:, right_hip]) / 2.
        #root = tf.expand_dims(root, 1)
        root = (joints[:, upper_left_hip] + joints[:, upper_right_hip] +
                joints[:, rear_left_hip] + joints[:, rear_right_hip]) / 4.

        joints = joints - root
    #else:
    #    print('Unknown skeleton type')
    #    import ipdb;
    #    ipdb.set_trace()

    curr_joints = joints[:-1]
    next_joints = joints[1:]
    loss = nn.MSELoss(reduce=None)

    return loss(curr_joints, next_joints)  # tf.losses.mean_squared_error(curr_joints, next_joints)


def init_pose(pred_Rs, init_pose, weights=None):
    """
    Should stay close to initial weights
    pred_Rs is N x 24 x 3 x 3
    init_pose is 72D, need to conver to Rodrigues (34, 35)
    """
    #init_Rs = batch_rodrigues(tf.reshape(init_pose, [-1, 3]))
    # init_Rs = tf.reshape(init_Rs, [-1, 24, 3, 3])
    init_pose = torch.reshape(init_pose, [-1, 3])
    # print('init pose size: ', init_pose.size())
    init_Rs = batch_rodrigues(init_pose)
    init_Rs = torch.reshape(init_Rs, [-1, 35, 3, 3])
    #RRt = tf.matmul(init_Rs, pred_Rs, transpose_b=True)
    # pred_Rs = torch.transpose(pred_Rs, 0, 1)
    # print('init Rs size: ', init_Rs.size())
    # print('pred Rs size: ', pred_Rs.size())
    RRt = None
    for i in range(0, len(init_Rs)):
        init_R_mat = init_Rs[i]
        pred_R_mat = pred_Rs[i]
        #print('init_R_mat size: ', init_R_mat.size())
        #print('pred_R_mat size: ', pred_R_mat.size())
        res = torch.unsqueeze(torch.matmul(init_R_mat, pred_R_mat), 0)
        if i == 0:
            #print('initial res size: ', res.size())
            RRt = res
        else:
            #print('before stack RRt size: ', RRt.size())
            #print('before stack res size: ', res.size())
            RRt = torch.cat((RRt, res), 0)


    # print('res size: ', res.size())
    # print('init R mat size: ', init_R_mat.size())
    print('RRt size: ', RRt.size())
    # RRt = torch.matmul(init_Rs, pred_Rs)
    traces = torch.zeros(RRt.size()[:-2])
    print('traces size: ', traces.size())
    for i in range(0, len(traces)):
        batch_mats = RRt[i]
        # print('batch mats size: ', batch_mats.size())
        traces[i] = batch_mats.diagonal(offset=0, dim1=-1,dim2=-2).sum(-1)
        # print('batch trace size: ', batch_trace.size())
    costheta = (traces - 1) / 2.
    target = torch.ones(costheta.size())  # tf.ones_like(costheta)
    if weights is None:
        weights = torch.ones(costheta.size()) # tf.ones_like(costheta)
    # not sure if this will keep gradients?
    return weighted_mse_loss(target, costheta, weights)  # tf.losses.mean_squared_error(target, costheta, weights=weights)


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def camera_smoothness(cams, scale_factors, offsets, img_size=224):
    # cams: [s, tx, ty]

    scales = cams[:, 0]
    actual_scales = scales * (1. / scale_factors)
    trans = cams[:, 1:]
    # pred trans + bbox top left corner / img_size
    actual_trans = ((trans + 1) * img_size * 0.5 + offsets) / img_size

    curr_scales = actual_scales[:-1]
    next_scales = actual_scales[1:]

    curr_trans = actual_trans[:-1]
    next_trans = actual_trans[1:]

    #scale_diff = tf.losses.mean_squared_error(curr_scales, next_scales)
    #trans_diff = tf.losses.mean_squared_error(curr_trans, next_trans)
    loss = nn.MSELoss(reduce=None)

    scale_diff = loss(curr_scales, next_scales)
    trans_diff = loss(curr_trans, next_trans)
    return scale_diff + trans_diff
