import os

import torch

from ..agents import make_hier_agent
from ..components.checkpointer import CheckpointHandler
from ..components.logger import Logger, WandBLogger, logger
from ..modules.replay_buffer import (HerReplayBuffer, ReplayBuffer,
                                     get_hier_buffer_samplers)
from ..modules.sampler import HierarchicalSampler
from ..utils.general_utils import (AttrDict, AverageMeter, Every, Timer, Until,
                                   set_seed_everywhere)
from ..utils.mpi import (mpi_gather_experience_successful_transitions,
                         mpi_gather_experience_transitions, mpi_sum,
                         mpi_gather_experience_rollots,
                         update_mpi_config)
from ..utils.rl_utils import RolloutStorage, init_demo_buffer, init_sc_buffer
from .sl_trainer import SkillLearningTrainer


class SkillChainingTrainer(SkillLearningTrainer):
    def _setup(self):
        self._setup_env()       # Environment
        self._setup_buffer()    # Relay buffer
        self._setup_agent()     # Agent
        self._setup_sampler()   # Sampler
        self._setup_logger()    # Logger
        self._setup_misc()      # MISC

        if self.is_chef:
            self.termlog.info('Setup done')

    def _setup_buffer(self):
        # Skill learning buffer -> HER dict replay buffer
        self.sl_buffer_samplers = get_hier_buffer_samplers(self.train_env, self.cfg.sl_agent.sampler)
        self.sl_buffer, self.sl_demo_buffer = AttrDict(), AttrDict()
        if self.cfg.agent.update_sl_agent:
            self.sl_buffer.update({subtask: HerReplayBuffer(
                buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params, batch_size=self.cfg.batch_size, 
                sampler=self.sl_buffer_samplers[subtask], T=self.env_params.subtask_steps[subtask]) for subtask in self.env_params.subtasks}
            )
            self.sl_demo_buffer.update({subtask: HerReplayBuffer(
                buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params, batch_size=self.cfg.batch_size, 
                sampler=self.sl_buffer_samplers[subtask], T=self.env_params.subtask_steps[subtask]) for subtask in self.env_params.subtasks}
            )
        
        # Skill chaining buffer -> Rollout state replay buffer
        self.sc_buffer = AttrDict({subtask : ReplayBuffer(
            obs_shape=self.env_params['obs'], action_shape=self.env_params['act_sc'], 
            capacity=self.cfg.replay_buffer_capacity, batch_size=self.cfg.batch_size, len_cond=self.env_params['len_cond']) for subtask in self.env_params.subtasks}
        )
        self.sc_demo_buffer = AttrDict({subtask : ReplayBuffer(
            obs_shape=self.env_params['obs'], action_shape=self.env_params['act_sc'], 
            capacity=self.cfg.replay_buffer_capacity, batch_size=self.cfg.batch_size, len_cond=self.env_params['len_cond']) for subtask in self.env_params.subtasks}
        )

    def _setup_agent(self):
        update_mpi_config(self.cfg)
        self.agent = make_hier_agent(self.env_params, self.sl_buffer_samplers, self.cfg)

    def _setup_sampler(self):
        self.train_sampler = HierarchicalSampler(self.train_env, self.agent, self.env_params, self.cfg)
        self.eval_sampler = HierarchicalSampler(self.eval_env, self.agent, self.env_params, self.cfg)

    def _setup_logger(self):
        if self.is_chef:
            exp_name = f"SC_{self.cfg.task}_{self.cfg.agent.sc_agent.name}_{self.cfg.agent.sl_agent.name}_seed{self.cfg.seed}"
            if self.cfg.postfix is not None:
                exp_name =  exp_name + '_' + self.cfg.postfix 
            if self.cfg.use_wb:
                self.wb = WandBLogger(exp_name=exp_name, project_name=self.cfg.project_name, entity=self.cfg.entity_name, \
                    path=self.work_dir, conf=self.cfg)
            else:
                self.wb = None
            self.logger = Logger(self.work_dir)
            self.termlog = logger
        else:
            self.wb, self.logger, self.termlog = None, None, None

    def _setup_misc(self):
        init_sc_buffer(self.cfg, self.sc_buffer, self.agent, self.env_params)
        init_sc_buffer(self.cfg, self.sc_demo_buffer, self.agent, self.env_params)
        
        if self.cfg.agent.update_sl_agent:
            for subtask in self.env_params.middle_subtasks:
                init_demo_buffer(self.cfg, self.sl_buffer[subtask], self.agent.sl_agent[subtask], subtask, False)
                init_demo_buffer(self.cfg, self.sl_demo_buffer[subtask], self.agent.sl_agent[subtask], subtask, False)

        if self.is_chef:
            self.model_dir = self.work_dir / 'model'
            self.model_dir.mkdir(exist_ok=True)
            for file in os.listdir(self.model_dir):
                os.remove(self.model_dir / file)

        self.device = torch.device(self.cfg.device)
        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0
        set_seed_everywhere(self.cfg.seed)
    
    def train(self):
        n_train_episodes = int(self.cfg.n_train_steps / self.env_params['max_timesteps'])
        n_eval_episodes = int(n_train_episodes / self.cfg.n_eval) * self.cfg.mpi.num_workers
        n_save_episodes = int(n_train_episodes / self.cfg.n_save) * self.cfg.mpi.num_workers
        n_log_episodes = int(n_train_episodes / self.cfg.n_log) * self.cfg.mpi.num_workers

        assert n_save_episodes >= n_eval_episodes
        if n_save_episodes % n_eval_episodes != 0:
            n_save_episodes = int(n_save_episodes / n_eval_episodes) * n_eval_episodes

        train_until_episode = Until(n_train_episodes)
        save_every_episodes = Every(n_save_episodes)
        eval_every_episodes = Every(n_eval_episodes)
        log_every_episodes = Every(n_log_episodes)
        seed_until_steps = Until(self.cfg.n_seed_steps)

        if self.is_chef:
            self.termlog.info('Starting training')
        while train_until_episode(self.global_episode):
            self._train_episode(log_every_episodes, seed_until_steps)

            if eval_every_episodes(self.global_episode):
                score = self.eval()

            if not self.cfg.dont_save and save_every_episodes(self.global_episode) and self.is_chef:
                filename =  CheckpointHandler.get_ckpt_name(self.global_episode)
                # TODO(tao): expose scoring metric
                CheckpointHandler.save_checkpoint({
                    'episode': self.global_episode,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                    'score': score,
                }, self.model_dir, filename)
                self.termlog.info(f'Save checkpoint to {os.path.join(self.model_dir, filename)}')

    def _train_episode(self, log_every_episodes, seed_until_steps):
        # sync network parameters across workers
        if self.use_multiple_workers:
            self.agent.sync_networks()

        self.timer.reset()
        batch_time = AverageMeter()
        ep_start_step = self.global_step
        metrics = None

        # collect experience and save to buffer
        rollout_storage = RolloutStorage()
        sc_episode, sl_episode, env_steps, _, _ = self.train_sampler.sample_episode(is_train=True, render=False)
        if self.use_multiple_workers:
            for subtask in sc_episode.sc_transitions.keys():
                transitions_batch = mpi_gather_experience_transitions(sc_episode.sc_transitions[subtask])
                # save to buffer
                self.sc_buffer[subtask].add_rollouts(transitions_batch)
                if self.cfg.agent.sc_agent.normalize:
                    self.agent.sc_agent.update_normalizer(transitions_batch, subtask)
                if self.cfg.agent.update_sl_agent:
                    for subtask in self.env_params.subtasks:
                        self.sl_buffer[subtask].store_episode(sl_episode[subtask])

        if self.use_multiple_workers:
            for subtask in sc_episode.sc_succ_transitions.keys():
                demo_batch = mpi_gather_experience_successful_transitions(sc_episode.sc_succ_transitions[subtask])
                # save to buffer
                self.sc_demo_buffer[subtask].add_rollouts(demo_batch)
        else:
            raise NotImplementedError
            #transitions_batch = sc_episode.sc_transitions

        # update status
        rollout_storage.append(sc_episode)
        rollout_status = rollout_storage.rollout_stats()
        self._global_step += int(mpi_sum(env_steps))
        self._global_episode += int(mpi_sum(1))

        # update policy
        if not seed_until_steps(ep_start_step) and self.is_chef:
            if not self.cfg.use_demo_buffer:
                metrics = self.agent.update(self.sc_buffer, self.sl_buffer)
            else:
                metrics = self.agent.update(self.sc_buffer, self.sl_buffer, self.sc_demo_buffer, self.sl_demo_buffer)
        if self.use_multiple_workers:
            self.agent.sync_networks()

        # log results
        if metrics is not None and log_every_episodes(self.global_episode) and self.is_chef:
            elapsed_time, total_time = self.timer.reset()
            batch_time.update(elapsed_time)
            togo_train_time = batch_time.avg * (self.cfg.n_train_steps - ep_start_step) / env_steps / self.cfg.mpi.num_workers

            self.logger.log_metrics(metrics, self.global_step, ty='train')
            with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                log('fps', env_steps / elapsed_time)
                log('total_time', total_time)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('ETA', togo_train_time)
            if self.cfg.use_wb:
                self.wb.log_outputs(metrics, None, log_images=False, step=self.global_step, is_train=True)

    def eval_ckpt(self):
        '''Eval checkpoint.'''
        CheckpointHandler.load_checkpoint(
            self.cfg.sc_ckpt_dir, self.agent, self.device, self.cfg.sc_ckpt_episode
        )
        
        eval_rollout_storage = RolloutStorage()
        violations = []
        not_violate_and_success = []
        following_dis_list = []

        for eval_ep in range(self.cfg.n_eval_episodes):
            episode, _, env_steps, num_violations, following_dis = self.eval_sampler.sample_episode(is_train=False, render=True, eval_ep=eval_ep, glob_ep=self.global_episode)
            eval_rollout_storage.append(episode)
            if num_violations > 0:
                violations.append(1)
            else:
                violations.append(0)
            if num_violations == 0 and episode['success'][-1]:
                not_violate_and_success.append(1)
            else:
                not_violate_and_success.append(0)
            following_dis_list.append(following_dis)
        rollout_status = eval_rollout_storage.rollout_stats()
        
        # Display average number of violations per episode
        print(
            f"Rate of violated episodes: {sum(violations) / len(violations)}")
        print(
            f"Rate of successful and no-violation episodes: {sum(not_violate_and_success) / len(not_violate_and_success)}")

        # Display average following distance per episode
        print(
            f"average following distance per episode: {sum(following_dis_list) / len(following_dis_list)}")

        if self.use_multiple_workers:
            rollout_status = mpi_gather_experience_rollots(rollout_status)
            for key, value in rollout_status.items():
                rollout_status[key] = value.mean()

        if self.is_chef:
            if self.cfg.use_wb:
                self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=self.global_step)
            with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode', self.global_episode)
                log('step', self.global_step)

        self.termlog.info(f'Successful rate: {rollout_status.avg_success_rate}')
        del eval_rollout_storage        
