from ..utils.general_utils import AttrDict, listdict2dictlist
from ..utils.rl_utils import ReplayCache, ReplayCacheGT
from mpi4py import MPI
import numpy as np
import torch

from cbf.cbf import CBF
from clf.clf import CLF
from surrol.utils.pybullet_utils import (
    get_link_pose,
)
from torchdiffeq import odeint
from scipy.spatial.transform import Rotation


class Sampler:
    """Collects rollouts from the environment using the given agent."""
    def __init__(self, env, agent, max_episode_len, config):
        self._env = env
        self._agent = agent
        self._max_episode_len = max_episode_len
        self.cfg = config

        self._obs = None
        self._episode_step = 0
        self._episode_cache = ReplayCacheGT(max_episode_len)
        
        self.device = torch.device(
            'cuda:' + str(0)
            if torch.cuda.is_available() else 'cpu'
        )
        
        self.CBF = CBF([6, 64, 42]).to(self.device)
        self.CBF.load_state_dict(torch.load(
            f"./cbf/saved_model/{self.cfg.task[0:-1]}0/combined/0/CBF10.pth"))
        self.CBF.eval()
        
        self.CLF = CLF([6, 64, 18]).to(self.device)
        self.CLF.load_state_dict(torch.load(
            f"./clf/saved_model/{self.cfg.task[0:-1]}0/combined/0/CLF10.pth"))
        self.CLF.eval()

        self.dcbf_constraint_type = int(self.cfg.task[-1])
        print(f'Constraint type is {self.dcbf_constraint_type}')

    def init(self):
        """Starts a new rollout. Render indicates whether output should contain image."""
        self._episode_reset()

    def sample_action(self, obs, is_train):
        return self._agent.get_action(obs, noise=is_train)
    
    def sample_episode(self, is_train, eval_ep=None, glob_ep=None, render=False):
        """Samples one episode from the environment."""
        self.init()
        episode, done = [], False
        images = []
        tt = torch.tensor([0., 0.1]).to(self.device)
        num_violations = 0
        while not done and self._episode_step < self._max_episode_len:
            action = self.sample_action(self._obs, is_train)
            
            if action is None:
                break
            
            if render:
                if self.cfg.render_three_views:
                    front_rgb_array, right_rgb_array, top_rgb_array = self._env.render_three_views('rgb_array')
                    render_obs = np.concatenate([front_rgb_array, right_rgb_array, top_rgb_array], axis=1)
                else:
                    render_obs = self._env.render('rgb_array')

                images.append(render_obs)
            
            
            # ===================== constraint test =====================
            # Display whether the tip of the psm has touch the obstacle or not
            # True : Collide
            # False: Safe
            violate_constraint = False
            
            if self.dcbf_constraint_type == 1:
                # Sphere constraint
                radius = 0.05
                constraint_center, _ = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                violate_constraint = self.CBF.constraint_valid(constraint_type=self.dcbf_constraint_type,
                                                               robot=self._obs['observation'][[0, 1, 2, 7, 8, 9]],
                                                               constraint_center=constraint_center,
                                                               radius=radius)
            if violate_constraint:
                num_violations += 1
                print(f'Episode {eval_ep:02}: warning: violate the constraint at episode step {self._episode_step}')
                
                
            # ===================== CBF =====================
            # Modify action to satisfy no-collision constraint
            if self.cfg.use_dcbf and self.dcbf_constraint_type != 0:
                with torch.no_grad():
                    x0 = torch.tensor(
                        self._obs['observation'][[0, 1, 2, 7, 8, 9]]).unsqueeze(0).to(self.device).float()

                    u0 = 0.05 * \
                        torch.tensor(action[[0, 1, 2, 5, 6, 7]]).unsqueeze(0).to(self.device).float()

                    x_dim = x0.shape[-1]
                    
                    cbf_out = self.CBF.net(x0)

                    fx = cbf_out[:, :x_dim]
                    gx = cbf_out[:, x_dim:]

                    if self.dcbf_constraint_type == 1:
                        modified_action = self.CBF.dCBF_sphere(x0, u0, fx, gx, constraint_center, radius)

                    # Check if action is modified by CBF
                    if (modified_action.cpu().numpy() == 0.05 * action[[0, 1, 2, 5, 6, 7]]).all():
                        isModified = False
                    else:
                        isModified = True
                    
                    # Remember to scale back the action before input into gym environment
                    action[[0, 1, 2, 5, 6, 7]] = modified_action.cpu().numpy() / 0.05
            
            
            # ===================== CLF =====================            
        
            if self.cfg.use_dclf and self.dcbf_constraint_type != 0 and isModified:
                assert self.cfg.use_dcbf
                with torch.no_grad():
                    # predicted next position given the modified action
                    self.CBF.u = modified_action
                    pred_next_position = odeint(self.CBF, x0, tt)[1, :, :]

                # ------------ Get desired orientation ------------
                # Use predicted next position and the critic to get the desired orientation

                # Get initial guess for orientation
                with torch.no_grad():
                    orn_x0 = torch.tensor(
                        self._obs['observation'][[3, 4, 5, 10, 11, 12]]).unsqueeze(0).to(self.device).float()
                    self.CLF.u = torch.tensor(action[[3, 8]].reshape(1, 2)).to(self.device).float()
                    update_orn = odeint(self.CLF, orn_x0, tt)[1, 0, :]

                for _ in range(10):
                    o = torch.tensor(self._obs['observation']).reshape(1, -1).cuda().float()
                    g = torch.tensor(self._obs['desired_goal']).reshape(1, -1).cuda().float()
                    o[:, [0, 1, 2, 7, 8, 9]] = pred_next_position
                    update_orn.requires_grad = True
                    o[:, [3, 4, 5, 10, 11, 12]] = update_orn

                    # Calculate gradient of the critic with respect to the orientation
                    input_tensor = self._agent._preproc_inputs(o, g, device='cuda')
                    predicted_next_action = self._agent.actor(input_tensor)
                    value = self._agent.critic_target(input_tensor, predicted_next_action)
                    value.backward()

                    update_grad = update_orn.grad.clone().detach()

                    # update the orientation with the gradient
                    step_size = 0.001
                    with torch.no_grad():
                        updated_orn = update_orn+update_grad*step_size

                    # test the updated orn
                    with torch.no_grad():
                        o[:, [3, 4, 5, 10, 11, 12]] = updated_orn
                        input_tensor = self._agent._preproc_inputs(o, g, device='cuda')
                        predicted_next_action = self._agent.actor(input_tensor)
                        value_new = self._agent.critic_target(input_tensor, predicted_next_action)
                        if value_new.item() > value.item():
                            update_orn = updated_orn.clone().detach()
                        else:
                            break
                desired_orn = update_orn.clone().detach().unsqueeze(0)
                
                # ------------use desired orientation------------
                with torch.no_grad():
                    orn_x0 = torch.tensor(
                        self._obs['observation'][[3, 4, 5, 10, 11, 12]]).unsqueeze(0).to(self.device).float()

                    if self.cfg.task[:-3] == "BiPegBoard":
                        orn_u0 = torch.tensor(action[[3, 8]]).unsqueeze(0).to(self.device).float()
                        orn_u0[:, 0] *= np.deg2rad(15)
                        orn_u0[:, 1] *= np.deg2rad(30)
                    else: # BiPegTransfer only!
                        orn_u0 = np.deg2rad(30) * \
                            torch.tensor(action[[3, 8]]).unsqueeze(0).to(self.device).float()
                    
                    clf_out = self.CLF.net(orn_x0)

                    fx = clf_out[:, :6]
                    gx = clf_out[:, 6:]

                    modified_orn = self.CLF.dCLF(orn_x0, desired_orn, orn_u0, fx, gx)

                    # Remember to scale back the action before input into gym environment
                    if self.cfg.task[:-3] == "BiPegBoard":
                        action[3] = modified_orn[0].cpu().numpy() / np.deg2rad(15)
                        action[8] = modified_orn[1].cpu().numpy() / np.deg2rad(30)
                    else:
                        action[[3, 8]] = modified_orn.cpu().numpy() / np.deg2rad(30)
            
            
            obs, reward, done, info = self._env.step(action)
            episode.append(AttrDict(
                reward=reward,
                success=info['is_success'],
                info=info
            ))
            self._episode_cache.store_transition(obs, action, done, info['gt_goal'])
            if render:
                episode[-1].update(AttrDict(image=render_obs))

            # update stored observation
            self._obs = obs
            self._episode_step += 1

        if render:
            images = np.array(images)
            np.save(f"images/{self.cfg.task}/glob_{glob_ep}_ep{eval_ep}_rank{MPI.COMM_WORLD.Get_rank()}.npy", arr=images)
            
        episode[-1].done = True     # make sure episode is marked as done at final time step
        rollouts = self._episode_cache.pop()
        assert self._episode_step == self._max_episode_len
        return listdict2dictlist(episode), rollouts, self._episode_step

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        self._episode_cache.store_obs(self._obs)

    def _reset_env(self):
        return self._env.reset()


class HierarchicalSampler(Sampler):
    """Collects experience batches by rolling out a hierarchical agent. Aggregates low-level batches into HL batch."""
    def __init__(self, env, agent, env_params, config):
        super().__init__(env, agent, env_params['max_timesteps'], config)

        self._env_params = env_params
        self._episode_cache = AttrDict(
            {subtask: ReplayCache(steps) for subtask, steps in env_params.subtask_steps.items()})
    
    def sample_episode(self, is_train, eval_ep=None, glob_ep=None, render=False):
        """Samples one episode from the environment."""
        self.init()
        sc_transitions = AttrDict({subtask: [] for subtask in self._env_params.subtasks})
        sc_succ_transitions = AttrDict({subtask: [] for subtask in self._env_params.subtasks})
        sc_episode, sl_episode, done, prev_subtask_succ = [], AttrDict(), False, AttrDict()
        images = []
        tt = torch.tensor([0., 0.1]).to(self.device)
        num_violations = 0
        while not done and self._episode_step < self._max_episode_len:
            agent_output = self.sample_action(self._obs, is_train, self._env.subtask)
            if self.last_sc_action is None:
                self._episode_cache[self._env.subtask].store_obs(self._obs)

            if render:
                if self.cfg.render_three_views:
                    front_rgb_array, right_rgb_array, top_rgb_array = self._env.render_three_views('rgb_array')
                    render_obs = np.concatenate([front_rgb_array, right_rgb_array, top_rgb_array], axis=1)
                else:
                    render_obs = self._env.render('rgb_array')
                    
                images.append(render_obs)
                
            if agent_output.is_sc_step:
                self.last_sc_action = agent_output.sc_action
                self.reward_since_last_sc = 0

            # ===================== constraint test =====================
            # Display whether the tip of the psm has touch the obstacle or not
            # True : Collide
            # False: Safe
            violate_constraint = False
            
            if self.dcbf_constraint_type in {1, 2}:
                # Sphere constraint
                sphere_radius = 0.05
                sphere_center, _ = get_link_pose(self._env.obj_ids['obstacle'][0], -1)
                violate_constraint = self.CBF.constraint_valid(constraint_type='sphere',
                                                               robot=self._obs['observation'][[0, 1, 2, 7, 8, 9]],
                                                               constraint_center=sphere_center,
                                                               radius=sphere_radius)
                # Cylinder constraint
                cylinder_center, cylinder_ori = get_link_pose(self._env.obj_ids['obstacle'][1], -1)
                cylinder_length = 0.05 * 5.
                cylinder_radius = 0.018 * 5.
                rot_matrix = Rotation.from_quat(np.array(cylinder_ori)).as_matrix()
                original_ori_vector = np.array([0, 0, 1]).reshape([3, 1])
                current_ori_vector = (rot_matrix @ original_ori_vector).reshape(-1).tolist()

                # psm1
                psm1 = self._obs['observation'][0:3]
                psm1_proj_vec = np.dot(current_ori_vector, np.array(psm1) - np.array(cylinder_center))
                
                if psm1_proj_vec ** 2 < (cylinder_length / 2) ** 2:
                    out = self.CBF.constraint_valid(
                        constraint_type='cylinder',
                        robot=psm1,
                        constraint_center=cylinder_center,
                        radius=cylinder_radius,
                        ori_vector=current_ori_vector
                    )
                    psm1_area = 1 if out else 2
                else:
                    psm1_area = 0
                
                if self._episode_step != 0:
                    violate_constraint = violate_constraint or (last_psm1_area + psm1_area == 3)

                last_psm1_area = psm1_area

                # psm2
                psm2 = self._obs['observation'][7:10]
                psm2_proj_vec = np.dot(current_ori_vector, np.array(psm2) - np.array(cylinder_center))
                
                if psm2_proj_vec ** 2 < (cylinder_length / 2) ** 2:
                    out = self.CBF.constraint_valid(
                        constraint_type='cylinder',
                        robot=psm2,
                        constraint_center=cylinder_center,
                        radius=cylinder_radius,
                        ori_vector=current_ori_vector
                    )
                    psm2_area = 1 if out else 2
                else:
                    psm2_area = 0

                if self._episode_step != 0:
                    violate_constraint = True if violate_constraint else (last_psm2_area + psm2_area == 3)

                last_psm2_area = psm2_area

            if violate_constraint:
                num_violations += 1
                print(f'Episode {eval_ep:02}: warning: violate the constraint at episode step {self._episode_step}')
            
            # ===================== CBF =====================
            # Modify action to satisfy no-collision constraint
            action = agent_output.sl_action
            
            if self.cfg.use_dcbf and self.dcbf_constraint_type != 0:
                with torch.no_grad():
                    x0 = torch.tensor(
                        self._obs['observation'][[0, 1, 2, 7, 8, 9]]).unsqueeze(0).to(self.device).float()

                    u0 = 0.05 * \
                        torch.tensor(action[[0, 1, 2, 5, 6, 7]]).unsqueeze(0).to(self.device).float()

                    x_dim = x0.shape[-1]
                    
                    cbf_out = self.CBF.net(x0)

                    fx = cbf_out[:, :x_dim]
                    gx = cbf_out[:, x_dim:]

                    if self.dcbf_constraint_type in {1, 2}:
                        modified_action = self.CBF.dCBF_sphere(x0, u0, fx, gx, sphere_center, sphere_radius)
                        if psm1_area + psm2_area > 0:
                            modified_action = self.CBF.dCBF_cylinder(
                                x0, modified_action, fx, gx, current_ori_vector, cylinder_center,
                                cylinder_radius, psm1_area, psm2_area)
                    # Check if action is modified by CBF
                    if (modified_action.cpu().numpy() == 0.05 * action[[0, 1, 2, 5, 6, 7]]).all():
                        isModified = False
                    else:
                        isModified = True
                    
                    # Remember to scale back the action before input into gym environment
                    action[[0, 1, 2, 5, 6, 7]] = modified_action.cpu().numpy() / 0.05
            
            # ===================== CLF =====================            
        
            if self.cfg.use_dclf and self.dcbf_constraint_type != 0 and isModified:
                assert self.cfg.use_dcbf
                with torch.no_grad():
                    # predicted next position given the modified action
                    self.CBF.u = modified_action
                    pred_next_position = odeint(self.CBF, x0, tt)[1, :, :]

                # ------------ Get desired orientation ------------
                # Use predicted next position and the critic to get the desired orientation

                # Get initial guess for orientation
                with torch.no_grad():
                    orn_x0 = torch.tensor(
                        self._obs['observation'][[3, 4, 5, 10, 11, 12]]).unsqueeze(0).to(self.device).float()
                    self.CLF.u = torch.tensor(action[[3, 8]].reshape(1, 2)).to(self.device).float()
                    update_orn = odeint(self.CLF, orn_x0, tt)[1, 0, :]

                for _ in range(10):
                    o = torch.tensor(self._obs['observation']).reshape(1, -1).cuda().float()
                    g = torch.tensor(self._obs['desired_goal']).reshape(1, -1).cuda().float()
                    o[:, [0, 1, 2, 7, 8, 9]] = pred_next_position
                    update_orn.requires_grad = True
                    o[:, [3, 4, 5, 10, 11, 12]] = update_orn

                    # Calculate gradient of the critic with respect to the orientation
                    input_tensor = self._agent.sl_agent[self._env.subtask]._preproc_inputs(o, g, device='cuda')
                    predicted_next_action = self._agent.sl_agent[self._env.subtask].actor(input_tensor)
                    value = self._agent.sl_agent[self._env.subtask].critic_target(input_tensor, predicted_next_action)
                    value.backward()

                    update_grad = update_orn.grad.clone().detach()

                    # update the orientation with the gradient
                    step_size = 0.001
                    with torch.no_grad():
                        updated_orn = update_orn+update_grad*step_size

                    # test the updated orn
                    with torch.no_grad():
                        o[:, [3, 4, 5, 10, 11, 12]] = updated_orn
                        input_tensor = self._agent.sl_agent[self._env.subtask]._preproc_inputs(o, g, device='cuda')
                        predicted_next_action = self._agent.sl_agent[self._env.subtask].actor(input_tensor)
                        value_new = self._agent.sl_agent[self._env.subtask].critic_target(input_tensor, predicted_next_action)
                        if value_new.item() > value.item():
                            update_orn = updated_orn.clone().detach()
                        else:
                            break
                desired_orn = update_orn.clone().detach().unsqueeze(0)
                
                # ------------use desired orientation------------
                with torch.no_grad():
                    orn_x0 = torch.tensor(
                        self._obs['observation'][[3, 4, 5, 10, 11, 12]]).unsqueeze(0).to(self.device).float()

                    if self.cfg.task[:-3] == "BiPegBoard":
                        orn_u0 = torch.tensor(action[[3, 8]]).unsqueeze(0).to(self.device).float()
                        orn_u0[:, 0] *= np.deg2rad(15)
                        orn_u0[:, 1] *= np.deg2rad(30)
                    else: # BiPegTransfer only!
                        orn_u0 = np.deg2rad(30) * \
                            torch.tensor(action[[3, 8]]).unsqueeze(0).to(self.device).float()
                    
                    clf_out = self.CLF.net(orn_x0)

                    fx = clf_out[:, :6]
                    gx = clf_out[:, 6:]

                    modified_orn = self.CLF.dCLF(orn_x0, desired_orn, orn_u0, fx, gx)

                    # Remember to scale back the action before input into gym environment
                    if self.cfg.task[:-3] == "BiPegBoard":
                        action[3] = modified_orn[0, 0].cpu().numpy() / np.deg2rad(15)
                        action[8] = modified_orn[0, 1].cpu().numpy() / np.deg2rad(30)
                    else:
                        action[[3, 8]] = modified_orn[0, :].cpu().numpy() / np.deg2rad(30)
            
            
            obs, reward, done, info = self._env.step(action)
            self.reward_since_last_sc += reward
            if info['subtask_done']:
                if not done:
                    # store skill-chaining transition
                    sc_transitions[info['subtask']].append(
                        [self.last_sc_obs, self.last_sc_action, self.reward_since_last_sc, obs['observation'], done, obs['desired_goal']])

                    if info['subtask_is_success']:
                        sc_succ_transitions[info['subtask']].append(
                            [self.last_sc_obs, self.last_sc_action, self.reward_since_last_sc, obs['observation'], done, obs['desired_goal']])
                    else:
                        sc_succ_transitions[info['subtask']].append([None])

                    # middle subtask 
                    self._episode_cache[self._env.subtask].store_obs(obs)
                    self._episode_cache[self._env.prev_subtasks[self._env.subtask]].\
                        store_transition(obs, agent_output.sl_action, True)      
                    self.last_sc_obs = obs['observation']
                else:
                    # terminal subtask
                    sc_transitions[info['subtask']] = []
                    sc_transitions[info['subtask']].append(
                        [self.last_sc_obs, self.last_sc_action, self.reward_since_last_sc, obs['observation'], done, obs['desired_goal']])
                    if info['subtask_is_success']:
                        sc_succ_transitions[info['subtask']].append(
                            [self.last_sc_obs, self.last_sc_action, self.reward_since_last_sc, obs['observation'], done, obs['desired_goal']])
                    else:
                        sc_succ_transitions[info['subtask']].append([None])
                    self._episode_cache[self._env.subtask].store_transition(obs, agent_output.sl_action, True)
                prev_subtask_succ[self._env.subtask] = info['subtask_is_success']
            else:
                self._episode_cache[self._env.subtask].store_transition(obs, agent_output.sl_action, False)
            
            sc_episode.append(AttrDict(
                reward=reward, 
                success=info['is_success'], 
                info=info))
            
            if render:
                sc_episode[-1].update(AttrDict(image=render_obs))

            # update stored observation
            self._obs = obs
            self._episode_step += 1

        if render:
            images = np.array(images)
            np.save(f"images/{self.cfg.task}/glob_{glob_ep}_ep{eval_ep}_rank{MPI.COMM_WORLD.Get_rank()}.npy", arr=images)
        
        # What is this condition for?
        # assert self._episode_step == self._max_episode_len
        for subtask in self._env_params.subtasks:
            if subtask not in prev_subtask_succ.keys():
                sl_episode[subtask] = self._episode_cache[subtask].pop()
                continue
            if prev_subtask_succ[subtask]:
                sl_episode[subtask] = self._episode_cache[subtask].pop()
            else:
                self._episode_cache[subtask].pop()

        sc_episode = listdict2dictlist(sc_episode)
        sc_episode.update(AttrDict(
            sc_transitions=sc_transitions,
            sc_succ_transitions=sc_succ_transitions)
        )
        
        return sc_episode, sl_episode, self._episode_step, num_violations

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        self.last_sc_obs, self.last_sc_action = self._obs['observation'], None  # stores observation when last hl action was taken
        self.reward_since_last_sc = 0   # accumulates the reward since the last HL step for HL transition

    def sample_action(self, obs, is_train, subtask):
        return self._agent.get_action(obs, subtask, noise=is_train)