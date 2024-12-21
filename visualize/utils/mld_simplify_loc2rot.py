import numpy as np
import os
import argparse

import torch
import smplx
import h5py
from tqdm import tqdm

from .joints2rots.smplify import SMPLify3D

from .joints2rots import config
from .utils import OneEuroFilter

import trimesh

class joints2smpl:

    def __init__(self, num_frames, device_id, cuda=True, num_smplify_iters=50):
        self.device = torch.device("cuda:" + str(device_id) if cuda else "cpu")
        # self.device = torch.device("cpu")
        self.batch_size = 1
        self.num_joints = 22  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = num_smplify_iters
        self.fix_foot = False
        print(config.SMPL_MODEL_DIR)
        smplmodel = smplx.create(config.SMPL_MODEL_DIR,
                                 model_type="smpl", gender="neutral", ext="pkl",
                                 batch_size=self.batch_size).to(self.device)
        self.smplmodel = smplmodel

        # ## --- load the mean pose as original ----
        self.smpl_mean_file = config.SMPL_MEAN_FILE

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                            batch_size=self.batch_size,
                            joints_category=self.joint_category,
                            num_iters=self.num_smplify_iters,
                            device=self.device)

    def joint2smpl(self, input_joints):
        # input joints [frames, 22, 3]
        file = h5py.File(self.smpl_mean_file, 'r')
        init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(
            self.device)
        init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(
            self.device)
        cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)


        _smplify = self.smplify # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(self.batch_size, 72).to(self.device)
        pred_betas = torch.zeros(self.batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(self.batch_size, 3).to(self.device)

        # run the whole seqs
        num_seqs = input_joints.shape[0]
        all_meshes = np.zeros([num_seqs, 6890, 3])


        for idx in range(num_seqs):
            keypoints_3d = input_joints[idx] # *1.2 #scale problem [check first]
            keypoints_3d = torch.Tensor(keypoints_3d).to(self.device).float().unsqueeze(dim=0)
            # print(keypoints_3d.shape)
            # exit()
            if idx == 0:
                pred_betas[0, :] = init_mean_shape
                pred_pose[0, :] = init_mean_pose
                pred_cam_t[0, :] = cam_trans_zero
            else:
                # print(init_mean_shape.shape, pre_betas.shape)
                pred_betas[0, :] = pre_betas
                pred_pose[0, :] = pre_pose
                pred_cam_t[0, :] = pre_cam_t


            if self.joint_category == "AMASS":
                confidence_input = torch.ones(self.num_joints)
                # make sure the foot and ankle
                if self.fix_foot:
                    confidence_input[7] = 1.5
                    confidence_input[8] = 1.5
                    confidence_input[10] = 1.5
                    confidence_input[11] = 1.5
            elif self.joint_category == "MMM":
                confidence_input = torch.ones(self.num_joints)
            else:
                print("Such category not settle down!")

            # print(pred_pose.shape, pred_betas.shape, pred_cam_t.shape, keypoints_3d.shape, )
            (
                new_opt_vertices,
                new_opt_joints,
                new_opt_pose,
                new_opt_betas,
                new_opt_cam_t,
                new_opt_joint_loss,
            ) = _smplify(
                pred_pose.detach(),
                pred_betas.detach(),
                pred_cam_t.detach(),
                keypoints_3d,
                conf_3d=confidence_input.to(self.device),
                # seq_ind=idx,
            )

            outputp = self.smplmodel(
                betas=new_opt_betas,
                global_orient=new_opt_pose[:, :3],
                body_pose=new_opt_pose[:, 3:],
                transl=new_opt_cam_t,
                return_verts=True,
            )
            mesh_p = trimesh.Trimesh(
                vertices=outputp.vertices.detach().cpu().numpy().squeeze(),
                faces=self.smplmodel.faces,
                process=False,
            )
            print(idx)
            all_meshes[idx] = mesh_p.vertices
            pre_betas = new_opt_betas
            pre_pose = new_opt_pose
            pre_cam_t = cam_trans_zero
            # mesh_p.export(ply_path)
            # param = {}
            # param["beta"] = new_opt_betas.detach().cpu().numpy()
            # param["pose"] = new_opt_pose.detach().cpu().numpy()
            # param["cam"] = new_opt_cam_t.detach().cpu().numpy()
            # joblib.dump(param, dir_save + "/" + "motion_%04d" %
            #             idx + ".pkl", compress=3)

        return all_meshes


    def joint2smpl_smooth(self, input_joints, min_cutoff, beta):
        # input joints [frames, 22, 3]
        file = h5py.File(self.smpl_mean_file, 'r')
        init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(
            self.device)
        init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(
            self.device)
        cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)

        _smplify = self.smplify  # if init_params is None else self.smplify_fast

        # run the whole seqs
        num_seqs = input_joints.shape[0]

        # num_seqs = 40

        all_pose = torch.zeros(num_seqs, 72).to(self.device)
        all_betas = torch.zeros(num_seqs, 10).to(self.device)
        all_cam_t = torch.zeros(num_seqs, 3).to(self.device)

        all_meshes = np.zeros([num_seqs, 6890, 3])

        for idx in range(num_seqs):
            keypoints_3d = input_joints[idx]  # *1.2 #scale problem [check first]
            keypoints_3d = torch.Tensor(keypoints_3d).to(self.device).float().unsqueeze(dim=0)
            # print(keypoints_3d.shape)
            # exit()
            if idx == 0:
                pred_betas = init_mean_shape
                pred_pose = init_mean_pose
                pred_cam_t = cam_trans_zero
            else:
                pred_betas = all_betas[idx-1:idx]
                pred_pose = all_pose[idx-1:idx]
                pred_cam_t = all_cam_t[idx-1:idx]

            if self.joint_category == "AMASS":
                confidence_input = torch.ones(self.num_joints)
                # make sure the foot and ankle
                if self.fix_foot:
                    confidence_input[7] = 1.5
                    confidence_input[8] = 1.5
                    confidence_input[10] = 1.5
                    confidence_input[11] = 1.5
            elif self.joint_category == "MMM":
                confidence_input = torch.ones(self.num_joints)
            else:
                print("Such category not settle down!")

            # print(pred_pose.shape, pred_betas.shape, pred_cam_t.shape, keypoints_3d.shape, )
            (
                new_opt_vertices,
                new_opt_joints,
                new_opt_pose,
                new_opt_betas,
                new_opt_cam_t,
                new_opt_joint_loss,
            ) = _smplify(
                pred_pose.detach(),
                pred_betas.detach(),
                pred_cam_t.detach(),
                keypoints_3d,
                conf_3d=confidence_input.to(self.device),
                # seq_ind=idx,
            )

            if idx == 0:
                one_euro_filter = OneEuroFilter(
                    torch.zeros_like(pred_pose[0]),
                    pred_pose[0],
                    min_cutoff=min_cutoff,
                    beta=beta,
                )
            else:
                t = torch.ones_like(new_opt_pose[0]) * idx
                pose = one_euro_filter(t, new_opt_pose[0])
                new_opt_pose[0] = pose

            outputp = self.smplmodel(
                betas=new_opt_betas,
                global_orient=new_opt_pose[:, :3],
                body_pose=new_opt_pose[:, 3:],
                transl=new_opt_cam_t,
                return_verts=True,
            )
            mesh_p = trimesh.Trimesh(
                vertices=outputp.vertices.detach().cpu().numpy().squeeze(),
                faces=self.smplmodel.faces,
                process=False,
            )
            # print(idx)
            all_meshes[idx] = mesh_p.vertices
            all_betas[idx] = new_opt_betas
            all_pose[idx] = new_opt_pose
            all_cam_t[idx] = cam_trans_zero

        return all_meshes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='Blender file or dir with blender files')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    simplify = joints2smpl(device_id=params.device, cuda=params.cuda)

    if os.path.isfile(params.input_path) and params.input_path.endswith('.npy'):
        simplify.npy2smpl(params.input_path)
    elif os.path.isdir(params.input_path):
        files = [os.path.join(params.input_path, f) for f in os.listdir(params.input_path) if f.endswith('.npy')]
        for f in files:
            simplify.npy2smpl(f)