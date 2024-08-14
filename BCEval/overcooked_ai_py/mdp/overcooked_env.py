import copy
import time

import cv2
import gym
import gymnasium
import numpy as np
import pygame
import tqdm

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    OvercookedGridworld,
)
from overcooked_ai_py.mdp.overcooked_trajectory import (
    DEFAULT_TRAJ_KEYS,
    EPISODE_TRAJ_KEYS,
    TIMESTEP_TRAJ_KEYS,
)
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelActionManager,
    MotionPlanner,
)
from overcooked_ai_py.utils import append_dictionaries, mean_and_std_err
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

DEFAULT_ENV_PARAMS = {"horizon": 400}

MAX_HORIZON = 1e10

GAP = 7

LOCAL_CH = {
    "↑":"north",
    "↓":"south",
    "←":"west",
    "→":"east",
}

class OvercookedEnv(object):
    """
    An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.

    E.g. of how to instantiate OvercookedEnv:
    > mdp = OvercookedGridworld(...)
    > env = OvercookedEnv.from_mdp(mdp, horizon=400)
    """

    #########################
    # INSTANTIATION METHODS #
    #########################

    def __init__(
        self,
        mdp_generator_fn,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=0,
        num_mdp=1,
        initial_info={},
    ):
        """
        mdp_generator_fn (callable):    A no-argument function that returns a OvercookedGridworld instance
        start_state_fn (callable):      Function that returns start state for the MDP, called at each environment reset
        horizon (int):                  Number of steps before the environment returns done=True
        mlam_params (dict):             params for MediumLevelActionManager
        info_level (int):               Change amount of logging
        num_mdp (int):                  the number of mdp if we are using a list of mdps
        initial_info (dict):            the initial outside information feed into the generator function

        TODO: Potentially make changes based on this discussion
        https://github.com/HumanCompatibleAI/overcooked_ai/pull/22#discussion_r416786847
        """
        assert callable(mdp_generator_fn), (
            "OvercookedEnv takes in a OvercookedGridworld generator function. "
            "If trying to instantiate directly from a OvercookedGridworld "
            "instance, use the OvercookedEnv.from_mdp method"
        )
        self.agents = ["Agent0", "Agent1"]
        self.num_mdp = num_mdp
        self.variable_mdp = num_mdp > 1
        self.mdp_generator_fn = mdp_generator_fn
        self.horizon = horizon
        self._mlam = None
        self._mp = None
        self.mlam_params = mlam_params
        self.start_state_fn = start_state_fn
        self.info_level = info_level
        self.reset(outside_info=initial_info)
        if self.horizon >= MAX_HORIZON and self.info_level > 0:
            print(
                "Environment has (near-)infinite horizon and no terminal states. \
                Reduce info level of OvercookedEnv to not see this message."
            )
        self.name = "overcooked"
        self.pre_env_state = None
        

    @property
    def mlam(self):
        if self._mlam is None:
            if self.info_level > 0:
                # print("Computing MediumLevelActionManager")
                pass
            self._mlam = MediumLevelActionManager.from_pickle_or_compute(
                self.mdp, self.mlam_params, force_compute=False
            )
        return self._mlam

    @property
    def mp(self):
        if self._mp is None:
            if self._mlam is not None:
                self._mp = self.mlam.motion_planner
            else:
                if self.info_level > 0:
                    print("Computing MotionPlanner")
                self._mp = MotionPlanner.from_pickle_or_compute(
                    self.mdp,
                    self.mlam_params["counter_goals"],
                    force_compute=False,
                )
        return self._mp

    @staticmethod
    def from_mdp(
        mdp,
        start_state_fn=None,
        horizon=MAX_HORIZON,
        mlam_params=NO_COUNTERS_PARAMS,
        info_level=1,
        num_mdp=None,
    ):
        """
        Create an OvercookedEnv directly from a OvercookedGridworld mdp
        rather than a mdp generating function.
        """
        assert isinstance(mdp, OvercookedGridworld)
        if num_mdp is not None:
            assert num_mdp == 1
        mdp_generator_fn = lambda _ignored: mdp
        env = OvercookedEnv(
            mdp_generator_fn=mdp_generator_fn,
            start_state_fn=start_state_fn,
            horizon=horizon,
            mlam_params=mlam_params,
            info_level=info_level,
            num_mdp=1,
        )
        env.layout_name = mdp.layout_name
        return env

    #####################
    # BASIC CLASS UTILS #
    #####################

    @property
    def env_params(self):
        """
        Env params should be thought of as all of the params of an env WITHOUT the mdp.
        Alone, env_params is not sufficent to recreate a copy of the Env instance, but it is
        together with mdp_params (which is sufficient to build a copy of the Mdp instance).
        """
        return {
            "start_state_fn": self.start_state_fn,
            "horizon": self.horizon,
            "info_level": self.info_level,
            "num_mdp": self.num_mdp,
        }

    def copy(self):
        # TODO: Add testing for checking that these util methods are up to date?
        return OvercookedEnv(
            mdp_generator_fn=self.mdp_generator_fn,
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
            info_level=self.info_level,
            num_mdp=self.num_mdp,
        )

    #############################
    # ENV VISUALIZATION METHODS #
    #############################

    def __repr__(self):
        """
        Standard way to view the state of an environment programatically
        is just to print the Env object
        """
        return self.mdp.state_string(self.state)

    def display_states(self, *states):
        old_state = self.state
        for s in states:
            self.state = s
            print(self)
        self.state = old_state

    def print_state_transition(
        self, a_t, r_t, env_info, fname=None, display_phi=False
    ):
        """
        Terminal graphics visualization of a state transition.
        """
        # TODO: turn this into a "formatting action probs" function and add action symbols too
        action_probs = [
            None
            if "action_probs" not in agent_info.keys()
            else list(agent_info["action_probs"])
            for agent_info in env_info["agent_infos"]
        ]

        action_probs = [
            None
            if player_action_probs is None
            else [round(p, 2) for p in player_action_probs[0]]
            for player_action_probs in action_probs
        ]

        if display_phi:
            state_potential_str = (
                "\nState potential = " + str(env_info["phi_s_prime"]) + "\t"
            )
            potential_diff_str = (
                "Δ potential = "
                + str(0.99 * env_info["phi_s_prime"] - env_info["phi_s"])
                + "\n"
            )  # Assuming gamma 0.99
        else:
            state_potential_str = ""
            potential_diff_str = ""

        output_string = "Timestep: {}\nJoint action taken: {} \t Reward: {} + shaping_factor * {}\nAction probs by index: {} {} {}\n{}\n".format(
            self.state.timestep,
            tuple(Action.ACTION_TO_CHAR[a] for a in a_t),
            r_t,
            env_info["shaped_r_by_agent"],
            action_probs,
            state_potential_str,
            potential_diff_str,
            self,
        )

        if fname is None:
            print(output_string)
        else:
            with open(fname, "a", encoding='utf-8') as f:
                print(output_string, file=f)

    ###################
    # BASIC ENV LOGIC #
    ###################

    def step(
        self, joint_action, joint_agent_action_info=None, display_phi=False
    ):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        if joint_agent_action_info is None:
            joint_agent_action_info = [{}, {}]
        next_state, mdp_infos = self.mdp.get_state_transition(
            self.state, joint_action, display_phi, self.mp
        )

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict(joint_agent_action_info, mdp_infos)

        if done:
            self._add_episode_info(env_info)

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])
        
        done = self.get_cooked_one()
        timestep_sparse_reward = self.get_reward_()
        
        return (next_state, timestep_sparse_reward, done, env_info)

    def get_reward_(self):
        temp = str(self).split('\n')
        
        if self.get_cooked_one():
            return 4
        # 统计 1 0 洋葱的个数
        # 汤未做完，奖励在 0-3 之间
        play_num = 0
        pot_num = 0
        for en_ in temp:
           if en_.strip() == "":
               continue
           else:
               row_ch = en_.split(' ')
               for ch in row_ch:
                   if len(ch.strip()) == 0:
                       continue
                   else:
                        if 'P' in ch:
                            pot_num = ch.count("ø")
                            if '✓' in ch:
                                pot_num += 1
                        elif '1' in ch and 'o' in ch:
                            play_num += 1
                        elif '0' in ch and 'o' in ch:
                            play_num += 1

        return min(3, play_num + pot_num)
        
    def lossless_state_encoding_mdp(self, state):
        """
        Wrapper of the mdp's lossless_encoding
        """
        return self.mdp.lossless_state_encoding(state, self.horizon)

    def featurize_state_mdp(self, state, num_pots=2):
        """
        Wrapper of the mdp's featurize_state
        """
        return self.mdp.featurize_state(state, self.mlam, num_pots=num_pots)

    def reset(self, regen_mdp=True, outside_info={}):
        """
        Resets the environment. Does NOT reset the agent.
        Args:
            regen_mdp (bool): gives the option of not re-generating mdp on the reset,
                                which is particularly helpful with reproducing results on variable mdp
            outside_info (dict): the outside information that will be fed into the scheduling_fn (if used), which will
                                 in turn generate a new set of mdp_params that is used to regenerate mdp.
                                 Please note that, if you intend to use this arguments throughout the run,
                                 you need to have a "initial_info" dictionary with the same keys in the "env_params"
        """
        if regen_mdp:
            self.mdp = self.mdp_generator_fn(outside_info)
            self._mlam = None
            self._mp = None
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()

        events_dict = {
            k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES
        }
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array(
                [0] * self.mdp.num_players
            ),
            "cumulative_shaped_rewards_by_agent": np.array(
                [0] * self.mdp.num_players
            ),
        }
        self.game_stats = {**events_dict, **rewards_dict}

    def is_done(self):
        """Whether the episode is over."""
        return self.state.timestep >= self.horizon or self.mdp.is_terminal(
            self.state
        )

    def potential(self, mlam, state=None, gamma=0.99):
        """
        Return the potential of the environment's current state, if no state is provided
        Otherwise return the potential of `state`
        args:
            mlam (MediumLevelActionManager): the mlam of self.mdp
            state (OvercookedState): the current state we are evaluating the potential on
            gamma (float): discount rate
        """
        state = state if state else self.state
        return self.mdp.potential_function(state, mp=self.mp, gamma=gamma)

    def _prepare_info_dict(self, joint_agent_action_info, mdp_infos):
        """
        The normal timestep info dict will contain infos specifc to each agent's action taken,
        and reward shaping information.
        """
        # Get the agent action info, that could contain info about action probs, or other
        # custom user defined information
        env_info = {
            "agent_infos": [
                joint_agent_action_info[agent_idx]
                for agent_idx in range(self.mdp.num_players)
            ]
        }
        # TODO: This can be further simplified by having all the mdp_infos copied over to the env_infos automatically
        env_info["sparse_r_by_agent"] = mdp_infos["sparse_reward_by_agent"]
        env_info["shaped_r_by_agent"] = mdp_infos["shaped_reward_by_agent"]
        env_info["phi_s"] = (
            mdp_infos["phi_s"] if "phi_s" in mdp_infos else None
        )
        env_info["phi_s_prime"] = (
            mdp_infos["phi_s_prime"] if "phi_s_prime" in mdp_infos else None
        )
        return env_info

    def _add_episode_info(self, env_info):
        env_info["episode"] = {
            "ep_game_stats": self.game_stats,
            "ep_sparse_r": sum(
                self.game_stats["cumulative_sparse_rewards_by_agent"]
            ),
            "ep_shaped_r": sum(
                self.game_stats["cumulative_shaped_rewards_by_agent"]
            ),
            "ep_sparse_r_by_agent": self.game_stats[
                "cumulative_sparse_rewards_by_agent"
            ],
            "ep_shaped_r_by_agent": self.game_stats[
                "cumulative_shaped_rewards_by_agent"
            ],
            "ep_length": self.state.timestep,
        }
        return env_info

    def _update_game_stats(self, infos):
        """
        Update the game stats dict based on the events of the current step
        NOTE: the timer ticks after events are logged, so there can be events from time 0 to time self.horizon - 1
        """
        self.game_stats["cumulative_sparse_rewards_by_agent"] += np.array(
            infos["sparse_reward_by_agent"]
        )
        self.game_stats["cumulative_shaped_rewards_by_agent"] += np.array(
            infos["shaped_reward_by_agent"]
        )

        for event_type, bool_list_by_agent in infos["event_infos"].items():
            # For each event type, store the timestep if it occurred
            event_occurred_by_idx = [int(x) for x in bool_list_by_agent]
            for idx, event_by_agent in enumerate(event_occurred_by_idx):
                if event_by_agent:
                    self.game_stats[event_type][idx].append(
                        self.state.timestep
                    )

    ####################
    # TRAJECTORY LOGIC #
    ####################

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display:
            print("Starting state\n{}".format(self))
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display:
                print(self)
            if done:
                break
        successor_state = self.state
        self.reset(False)
        return successor_state, done

    def run_agents(
        self,
        agent_pair,
        include_final_state=False,
        display=False,
        dir=None,
        display_phi=False,
        display_until=np.Inf,
    ):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t, info_t).
        """
        assert (
            self.state.timestep == 0
        ), "Did not reset environment before running agents"
        trajectory = []
        done = False
        # default is to not print to file
        fname = None

        if dir != None:
            fname = dir + "/roll_out_" + str(time.time()) + ".txt"
            with open(fname, "w+", encoding='utf-8') as f:
                print(self, file=f)
        while not done:
            s_t = self.state

            # Getting actions and action infos (optional) for both agents
            joint_action_and_infos = agent_pair.joint_action(s_t)
            a_t, a_info_t = zip(*joint_action_and_infos)
            assert all(a in Action.ALL_ACTIONS for a in a_t)
            assert all(type(a_info) is dict for a_info in a_info_t)

            s_tp1, r_t, done, info = self.step(a_t, a_info_t, display_phi)
            trajectory.append((s_t, a_t, r_t, done, info))

            if display and self.state.timestep < display_until:
                self.print_state_transition(a_t, r_t, info, fname, display_phi)

        assert len(trajectory) == self.state.timestep, "{} vs {}".format(
            len(trajectory), self.state.timestep
        )

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True, None))

        total_sparse = sum(
            self.game_stats["cumulative_sparse_rewards_by_agent"]
        )
        total_shaped = sum(
            self.game_stats["cumulative_shaped_rewards_by_agent"]
        )
        return (
            np.array(trajectory, dtype=object),
            self.state.timestep,
            total_sparse,
            total_shaped,
        )

    def get_rollouts(
        self,
        agent_pair,
        num_games,
        display=False,
        dir=None,
        final_state=False,
        display_phi=False,
        display_until=np.Inf,
        metadata_fn=None,
        metadata_info_fn=None,
        info=True,
    ):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories.

        Returning excessive information to be able to convert trajectories to any required format
        (baselines, stable_baselines, etc)

        metadata_fn returns some metadata information computed at the end of each trajectory based on
        some of the trajectory data.

        NOTE: this is the standard trajectories format used throughout the codebase
        """
        trajectories = {k: [] for k in DEFAULT_TRAJ_KEYS}
        metadata_fn = (lambda x: {}) if metadata_fn is None else metadata_fn
        metadata_info_fn = (
            (lambda x: "") if metadata_info_fn is None else metadata_info_fn
        )
        range_iterator = (
            tqdm.trange(num_games, desc="", leave=True)
            if info
            else range(num_games)
        )
        for i in range_iterator:
            agent_pair.set_mdp(self.mdp)

            rollout_info = self.run_agents(
                agent_pair,
                display=display,
                dir=dir,
                include_final_state=final_state,
                display_phi=display_phi,
                display_until=display_until,
            )
            (
                trajectory,
                time_taken,
                tot_rews_sparse,
                _tot_rews_shaped,
            ) = rollout_info
            obs, actions, rews, dones, infos = (
                trajectory.T[0],
                trajectory.T[1],
                trajectory.T[2],
                trajectory.T[3],
                trajectory.T[4],
            )
            trajectories["ep_states"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_infos"].append(infos)
            trajectories["ep_returns"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)
            trajectories["metadatas"].append(metadata_fn(rollout_info))

            # we do not need to regenerate MDP if we are trying to generate a series of rollouts using the same MDP
            # Basically, the FALSE here means that we are using the same layout and starting positions
            # (if regen_mdp == True, resetting will call mdp_gen_fn to generate another layout & starting position)
            self.reset(regen_mdp=False)
            agent_pair.reset()

            if info:
                mu, se = mean_and_std_err(trajectories["ep_returns"])
                description = "Avg rew: {:.2f} (std: {:.2f}, se: {:.2f}); avg len: {:.2f}; ".format(
                    mu,
                    np.std(trajectories["ep_returns"]),
                    se,
                    np.mean(trajectories["ep_lengths"]),
                )
                description += metadata_info_fn(trajectories["metadatas"])
                range_iterator.set_description(description)
                range_iterator.refresh()

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}

        # Merging all metadata dictionaries, assumes same keys throughout all
        trajectories["metadatas"] = append_dictionaries(
            trajectories["metadatas"]
        )

        # TODO: should probably transfer check methods over to Env class
        from overcooked_ai_py.agents.benchmarking import AgentEvaluator

        AgentEvaluator.check_trajectories(trajectories, verbose=info)
        return trajectories

    ####################
    # TRAJECTORY UTILS #
    ####################

    @staticmethod
    def get_discounted_rewards(trajectories, gamma):
        rews = trajectories["ep_rewards"]
        horizon = rews.shape[1]
        return OvercookedEnv._get_discounted_rewards_with_horizon(
            rews, gamma, horizon
        )

    @staticmethod
    def _get_discounted_rewards_with_horizon(rewards_matrix, gamma, horizon):
        rewards_matrix = np.array(rewards_matrix)
        discount_array = [gamma**i for i in range(horizon)]
        rewards_matrix = rewards_matrix[:, :horizon]
        discounted_rews = np.sum(rewards_matrix * discount_array, axis=1)
        return discounted_rews

    @staticmethod
    def get_agent_infos_for_trajectories(trajectories, agent_idx):
        """
        Returns a dictionary of the form
        {
            "[agent_info_0]": [ [episode_values], [], ... ],
            "[agent_info_1]": [ [], [], ... ],
            ...
        }
        with as keys the keys returned by the agent in it's agent_info dictionary

        NOTE: deprecated
        """
        agent_infos = []
        for traj_idx in range(len(trajectories["ep_lengths"])):
            ep_infos = trajectories["ep_infos"][traj_idx]
            traj_agent_infos = [
                step_info["agent_infos"][agent_idx] for step_info in ep_infos
            ]

            # Append all dictionaries together
            traj_agent_infos = append_dictionaries(traj_agent_infos)
            agent_infos.append(traj_agent_infos)

        # Append all dictionaries together once again
        agent_infos = append_dictionaries(agent_infos)
        agent_infos = {k: np.array(v) for k, v in agent_infos.items()}
        return agent_infos

    @staticmethod
    def proportion_stuck_time(trajectories, agent_idx, stuck_time=3):
        """
        Simple util for calculating a guess for the proportion of time in the trajectories
        during which the agent with the desired agent index was stuck.

        NOTE: deprecated
        """
        stuck_matrix = []
        for traj_idx in range(len(trajectories["ep_lengths"])):
            stuck_matrix.append([])
            obs = trajectories["ep_states"][traj_idx]
            for traj_timestep in range(
                stuck_time, trajectories["ep_lengths"][traj_idx]
            ):
                if traj_timestep >= stuck_time:
                    recent_states = obs[
                        traj_timestep - stuck_time : traj_timestep + 1
                    ]
                    recent_player_pos_and_or = [
                        s.players[agent_idx].pos_and_or for s in recent_states
                    ]

                    if len({item for item in recent_player_pos_and_or}) == 1:
                        # If there is only one item in the last stuck_time steps, then we classify the agent as stuck
                        stuck_matrix[traj_idx].append(True)
                    else:
                        stuck_matrix[traj_idx].append(False)
                else:
                    stuck_matrix[traj_idx].append(False)
        return stuck_matrix

    # 可选动作格式
    ''', the details are as follows:
1. when the agent is in front of an onion or dish and the agent has no items to pick up, the action will pick up the onion or dish. 
2. When there is an empty table in front of the agent and the agent has an item in his hand, the item is placed on the table.
3. when the agent has an ingredient in his hand and the agent is in front of a cooking table and not placed 3 ingredients, the agent places the ingredient in his hand on the cooking table. 
4. when the agent has no item in his hand and the cooking table in front of the agent has ingredient, the action will make the cooking table start cooking soup. 
5. When the soup is cooked in front of the agent and the agent is holding the dish, the action puts the soup into the dish.
6. when the agent is holding the dish in his hand and is in front of the service desk, the agent delivers the soup and receives the reward for making the soup.
Each agent has a maximum of one object in their hand, and can only pick up another object when they put it down'''
    def get_task_prompt(self, agent_type:str="readJ"):
        # ingredient = self.mdp.recipe_config['all_orders'][0]['ingredients'][0]
        action_prompt = f"""
[Action Options]
1) north, move one step north.
2) south, move one step south.
3) east, move one step east.
4) west, move one step west.
5) stay, do nothing.
6) interact, interact with an object."""
        act = self.get_action_example(agent_type=agent_type)
        return f'{action_prompt}\n{act}'
        # return action_prompt
    
    def get_action_example(self, agent_type:str="readJ"):
        assert agent_type in ["readJ","roco","cent","react","reflexion","mind"], f"agent type should be in ['readJ','roco','cent','react','reflexion','mind']"
        if agent_type =="cent":
            example = """[Action Output Instruction]
Think step-by-step about your plan and output your idea, and then output 'EXECUTE\n', then give exactly one action per agent, put each on a new line.
Example#1:EXECUTE\nNAME Agent0 ACTION north\nNAME Agent1 ACTION interact
Example#2:EXECUTE\nNAME Agent0 ACTION east\nNAME Agent1 ACTION stay"""
        elif agent_type =="react":
            example = """[Action Output Instruction]
Think step-by-step about your plan and output your idea, and then output 'EXECUTE\n', then give exactly one action per agent, put each on a new line.
Example#1:Think: Since agent1 is holding the plate facing the cooking station, the soup is already cooked, agent1 should use the interact action to serve the soup, and agent0 should grab the top onion.\nAction: EXECUTE\nNAME Agent0 ACTION north\nNAME Agent1 ACTION interact
Example#2:Think: The soup is not ready yet and agent1 has the plate in his hand, so agent1 is not moving at this turn and agent0 should go to the nearest place to get the onion.\nAction: EXECUTE\nNAME Agent0 ACTION east\nNAME Agent1 ACTION stay"""
        else:
            example = """[Action Output Instruction]
Think step-by-step about your plan and output your idea, and then output 'EXECUTE\n', then give exactly one action per agent, put each on a new line.
Example#1: Since agent1 is holding the plate facing the cooking station, the soup is already cooked, agent1 should use the interact action to serve the soup, and agent0 should grab the top onion.\nEXECUTE\nNAME Agent0 ACTION north\nNAME Agent1 ACTION interact
Example#2: The soup is not ready yet and agent1 has the plate in his hand, so agent1 is not moving at this turn and agent0 should go to the nearest place to get the onion.\nEXECUTE\nNAME Agent0 ACTION east\nNAME Agent1 ACTION stay"""
        return example
    
    # 环境说明 有几个agent 有哪些原料 recipe信息
    def get_env_prompt(self):
        terrian_list = self.mdp.terrain_mtx
        row_num = len(terrian_list)
        col_num = len(terrian_list[0])
        terrian_str = '\n'.join([''.join(terrian_row) for terrian_row in terrian_list])
        recipe_len = self.mdp.recipe_config['num_items_for_soup']
        # print(self.mdp.recipe_config.keys())
        # exit(0)
        cook_time = self.mdp.recipe_config['cook_time']
        ingredient = self.mdp.recipe_config['all_orders'][0]['ingredients'][0]
        
        
        format_mean = '''The letter X stands for table, P for cooking station, O and o stand for onion, D and d for plates, and S for service desk. 
When the onion or dish is on the table or being held by agent, an o or d will be added after its corresponding character.
When the onion is placed on the cooking table, it will be denoted as p{ø, p{øø means that there are two onions on the cooking table.
And when the cooking table cooks the soup, it will show how long it has been cooked, such as p{ø20 means that it has been cooked in 20 time steps. 
The numbers 1 and 0 represent the agent, and the direction arrow ↑ ↓ ← → represents the direction the agent is facing.Each object occupies a grid size, and the agent moves one grid distance at a time.'''
        
        env_prompt = f"""[Task Information]
This is overcooked environment. Two agents need to collaborate to cook soup and deliver the cooked soup to the service desk to earn a reward. 
Each soup needs {recipe_len} {ingredient}s, pick up {recipe_len} {ingredient}s and put them into the cooking table for cooking, 
when the cooking table shows the number of {cook_time}, the soup is finished, and you need to take a plate of soup and deliver 
it to the service desk, which can get a bowl of soup reward, you need to think and give the actions of two agents, to ensure that 
after 30 interactions can get a high enough reward.
[Layout Information]
The size of the room is a {row_num} × {col_num} grid, and the overall layout is:
{terrian_str}
[Character meaning]
{format_mean}"""
        return env_prompt
    
    # 环境的状态
    def get_env_state_prompt(self):
        # 环境当前的状态 + 烹饪台的状态
        temp = str(self).split('\n')
        l = len(temp)
        env_str = []
        # env_str = temp
        for i,st in enumerate(temp):
        # 去最后空行
            if st.strip() == "" and i==l-1:
                continue
            env_str.append(''.join(st))
        env_str = '\n'.join(env_str)
        self.state_dict = {
            "p_local": set(),
            "p_o_num": 0,
            "p_cooked_time":0,
            "p_cook_finish":False,
            "0_local": [],
            "1_local": [],
            "0_direction":"",
            "1_direction":"",
            "0_type": "",
            "1_type": "",
            "O_local": set(),
            "D_local": set(),
            "S_local": set(),
            "o_local": set(),
        }
        row = 0
        for en_ in temp:
        #    print(len(en_))
        #    print(en_.count(' '))   
        #    exit(0)   
           if en_.strip() == "":
               continue
           else:
               row_ch = en_.split(' ')
            #    col = 0
               for ch in row_ch:
                   if len(ch.strip()) == 0:
                       continue
                   else:
                    if 'P' in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "P":
                                col = int(idx/GAP)
                                self.state_dict["p_local"].add((row,col))
                        # self.state_dict["p_local"] = [row,col]
                        self.state_dict["p_o_num"] = ch.count("ø")
                        self.state_dict["p_cook_finish"] = "✓" in ch
                        time_temp = ch.split('ø')[-1].strip()
                        if time_temp.isdigit():
                            time_temp = int(time_temp)
                        else:
                            time_temp = 0
                        self.state_dict["p_cooked_time"] = time_temp   
                    elif "0" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "0":
                                col = int(idx/GAP)
                                self.state_dict["0_local"].append((row,col))
                        # self.state_dict["0_local"] = [row,col]
                        self.state_dict["0_direction"] = LOCAL_CH[ch.split("0")[0].strip()]
                        if "d" in ch:
                            self.state_dict["0_type"] = "plate"
                        elif "o" in ch:
                            self.state_dict["0_type"] = "onion"
                        elif "✓" in ch:
                            self.state_dict["0_type"] = "soup"
                        else:
                            self.state_dict["0_type"] = "None"
                    elif "1" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "1":
                                col = int(idx/GAP)
                                self.state_dict["1_local"].append((row,col))
                        # self.state_dict["1_local"] = [row,col]
                        self.state_dict["1_direction"] = LOCAL_CH[ch.split("1")[0].strip()]
                        if "d" in ch:
                            self.state_dict["1_type"] = "plate"
                        elif "o" in ch:
                            self.state_dict["1_type"] = "onion"
                        elif "✓" in ch:
                            self.state_dict["1_type"] = "soup"
                        else:
                            self.state_dict["1_type"] = "None"
                    elif "O" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "O":
                                col = int(idx/GAP)
                                self.state_dict["O_local"].add((row,col))
                    elif "D" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "D":
                                col = int(idx/GAP)
                                self.state_dict["D_local"].add((row,col))
                        # self.state_dict["D_local"].append([row,col])
                    elif "S" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "S":
                                col = int(idx/GAP)
                                self.state_dict["S_local"].add((row,col))
                        # self.state_dict["S_local"].append([row,col])
                    elif "o" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "o":
                                col = int(idx/GAP)
                                self.state_dict["o_local"].add((row,col))
                    #    self.state_dict["o_local"].append([row,col])
                    
                #    col += 1
               row += 1
                
                
        # pos_information
        env_pos_information, _ = self.parser_state_dict()
        
        env_state = env_str + env_pos_information
        return env_state

    def parser_state_dict(self):
        if self.state_dict["p_cook_finish"]:
            cook_state = "soup has been cooked"
        elif not self.state_dict["p_cook_finish"] and self.state_dict["p_cooked_time"] != 0:
            cook_state = "soup has been cooked for {} steps".format(self.state_dict["p_cooked_time"])
        else:
            cook_state = "soup has not yet begun to cook"
        p_local = ""
        for pos in self.state_dict["p_local"]:
            p_local += f"{pos} "
        p_local = p_local.strip()
        p_state = f"Cook station local {p_local}, there are {self.state_dict['p_o_num']} onions on it, {cook_state}\n"
        O_state = "Onions local: "
        for pos in self.state_dict["O_local"]:
            O_state += f"{pos} "
        O_state += '\n'
        D_state = "Dishes local: "
        for pos in self.state_dict["D_local"]:
            D_state += f"{pos}"
        D_state += '\n'
        S_state = "Server desk local: "
        for pos in self.state_dict["S_local"]:
            S_state += f"{pos}"
        # S_state += '\n'
        o_state = "\nThe following coordinates have Onions on the table: "
        for pos in self.state_dict["o_local"]:
            o_state += f"{pos}"
        o_state += '\n'
        env_pos_information = f"{p_state}{O_state}{D_state}{S_state}{o_state if len(self.state_dict['o_local']) else ''}"
        
        Player_0_state = "Agent0 local: {}, direction: {}, hold: {}\n".format(self.state_dict["0_local"], self.state_dict["0_direction"], self.state_dict["0_type"] if self.state_dict["0_type"]!="None" else "nothing")
        
        Player_1_state = "Agent1 local: {}, direction: {}, hold: {}\n".format(self.state_dict["1_local"], self.state_dict["1_direction"], self.state_dict["1_type"] if self.state_dict["1_type"]!="None" else "nothing")
        
        player_state = Player_0_state + Player_1_state
        
        return env_pos_information, player_state
        
    # agent的状态
    def get_agent_state_prompt(self):
        # 两个玩家的状态
        # 环境当前的状态 + 烹饪台的状态
        temp = str(self).split('\n')
        env_str = []
        
        for i,st in enumerate(temp):
        # 去空行
            if st.strip() == "":
                continue
            env_str.append(''.join(st))
        env_str = '\n'.join(env_str)
        self.state_dict = {
            "p_local": set(),
            "p_o_num": 0,
            "p_cooked_time":0,
            "p_cook_finish":False,
            "0_local": [],
            "1_local": [],
            "0_direction":"",
            "1_direction":"",
            "0_type": "",
            "1_type": "",
            "O_local": set(),
            "D_local": set(),
            "S_local": set(),
            "o_local": set(),
        }
        row = 0
        for en_ in temp:
        #    print(len(en_))
        #    print(en_.count(' '))   
        #    exit(0)   
           if en_.strip() == "":
               continue
           else:
               row_ch = en_.split(' ')
            #    col = 0
               for ch in row_ch:
                   if len(ch.strip()) == 0:
                       continue
                   else:
                    if 'P' in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "P":
                                col = int(idx/GAP)
                                self.state_dict["p_local"].add((row,col))
                        # self.state_dict["p_local"] = [row,col]
                        self.state_dict["p_o_num"] = ch.count("ø")
                        self.state_dict["p_cook_finish"] = "✓" in ch
                        time_temp = ch.split('ø')[-1].strip()
                        if time_temp.isdigit():
                            time_temp = int(time_temp)
                        else:
                            time_temp = 0
                        self.state_dict["p_cooked_time"] = time_temp   
                    elif "0" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "0":
                                col = int(idx/GAP)
                                self.state_dict["0_local"]=(row,col)
                        # self.state_dict["0_local"] = [row,col]
                        self.state_dict["0_direction"] = LOCAL_CH[ch.split("0")[0].strip()]
                        if "d" in ch:
                            self.state_dict["0_type"] = "plate"
                        elif "o" in ch:
                            self.state_dict["0_type"] = "onion"
                        elif "✓" in ch:
                            self.state_dict["0_type"] = "soup"
                        else:
                            self.state_dict["0_type"] = "None"
                    elif "1" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "1":
                                col = int(idx/GAP)
                                self.state_dict["1_local"]=(row,col)
                        # self.state_dict["1_local"] = [row,col]
                        self.state_dict["1_direction"] = LOCAL_CH[ch.split("1")[0].strip()]
                        if "d" in ch:
                            self.state_dict["1_type"] = "plate"
                        elif "o" in ch:
                            self.state_dict["1_type"] = "onion"
                        elif "✓" in ch:
                            self.state_dict["1_type"] = "soup"
                        else:
                            self.state_dict["1_type"] = "None"
                    elif "O" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "O":
                                col = int(idx/GAP)
                                self.state_dict["O_local"].add((row,col))
                    elif "D" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "D":
                                col = int(idx/GAP)
                                self.state_dict["D_local"].add((row,col))
                        # self.state_dict["D_local"].append([row,col])
                    elif "S" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "S":
                                col = int(idx/GAP)
                                self.state_dict["S_local"].add((row,col))
                        # self.state_dict["S_local"].append([row,col])
                    elif "o" in ch:
                        idx = 0
                        for idx in range(len(en_)):
                            if en_[idx] == "o":
                                col = int(idx/GAP)
                                self.state_dict["o_local"].add((row,col))
                    #    self.state_dict["o_local"].append([row,col])
                    
                #    col += 1
               row += 1
                
                
               
        # pos_information
        _, agent_state = self.parser_state_dict()
        
        agent_state_ =  agent_state
        return agent_state_

    def get_read_prompt(self):
        action_pro = self.get_task_prompt()
        action_exa = self.get_action_example(agent_type="readJ")
        task_pro = self.get_env_prompt()
        env_pro = self.get_env_state_prompt()
        agent_pro = self.get_agent_state_prompt()
        return f'[Current Env state]:\n{env_pro}\n{agent_pro}', task_pro
    
    def get_preety(self):
        temp = str(self).split('\n')
        env_str = []
        
        for i,st in enumerate(temp):
        # 去空行
            if st.strip() == "":
                continue
            env_str.append(''.join(st))
        env_str = '\n'.join(env_str)
        return env_str

    def get_cooked_one(self):
        
        if self.pre_env_state is None:
            self.pre_env_state = self.get_env_state_prompt()
            return False
        else:
           cur_env_state = self.get_env_state_prompt()
           if 'øø✓' in self.pre_env_state and 'øø✓' not in cur_env_state:
               return True
           else:
               self.pre_env_state = cur_env_state
               return False 
    
    
from pettingzoo.utils.env import ParallelEnv

from overcooked_ai_py.agents.agent import AgentPair


class OvercookedEnvPettingZoo(ParallelEnv):
    def __init__(self, base_env, agents):
        """
        base_env: OvercookedEnv
        agents: AgentPair

        Example creating a PettingZoo env from a base_env:

        mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
        base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
        agent_pair = load_agent_pair("path/to/checkpoint", "ppo", "ppo")
        env = OvercookedEnvPettingZoo(base_env, agent_pair)

        """
        # we need agent-dependent observation space, and the best way to do it is just to include an agentPair
        assert isinstance(
            agents, AgentPair
        ), "agents must be an AgentPair object"

        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = ["agent_0", "agent_1"]
        self.agent_map = {"agent_0": agents.a0, "agent_1": agents.a1}
        self.base_env = base_env
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.agents
        }
        self.action_spaces = {
            agent: gymnasium.spaces.Discrete(len(Action.ALL_ACTIONS))
            for agent in self.agents
        }
        # this is the AgentPair object
        self.reset()

    import functools

    # we want to return the same space object every time
    @functools.lru_cache(maxsize=2)
    def observation_space(self, agent):
        # the observation can be different for each agent
        agent = self.agent_map[agent]
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = agent.featurize(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.zeros(obs_shape)
        return gymnasium.spaces.Box(low, high, dtype=np.float32)

    # we want to return the same space object every time
    @functools.lru_cache(maxsize=1)
    def action_space(self, agent):
        # the action space is the same for each agent
        return gymnasium.spaces.Discrete(len(Action.ALL_ACTIONS))

    def step(self, joint_action):
        joint_action = [
            Action.ALL_ACTIONS[joint_action[agent]] for agent in joint_action
        ]
        
        obs, reward, done, info = self.base_env.step(joint_action)
        # https://gymnasium.farama.org/content/basic_usage/
        # we have no early termination condition in this env, and the environment only terminates when the time horizon is reached
        # therefore the terminated is always False, and we set truncated to done
        terminated = False
        truncated = done

        def create_dict(value):
            """
            Each agent should have the same reward, terminated, truncated, info
            """
            return {agent: value for agent in self.agents}

        def create_obs_dict(obs):
            """
            Observation is potentially different for each agent
            """
            return {
                agent: self.agent_map[agent].featurize(obs)
                for agent in self.agents
            }

        obs = create_obs_dict(obs)
        reward = create_dict(reward)
        terminated = create_dict(terminated)
        truncated = create_dict(truncated)
        info = create_dict(info)
        if done:
            self.agents = []
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the embedded OvercookedEnv envrionment to the starting state
        """
        self.base_env.reset()
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        # when an environment terminates/truncates, PettingZoo wants all agents removed, so during reset we re-add them
        self.agents = self.possible_agents[:]
        # return the obsevations as dict
        obs_dict = {
            agent: self.agent_map[agent].featurize(dummy_state)[0]
            for agent in self.agents
        }
        self.pre_env_state = None
        return obs_dict, None

    def render(self, mode="human", close=False):
        pass


class Overcooked(gym.Env):
    """
    Wrapper for the Env class above that is SOMEWHAT compatible with the standard gym API.
    Why only somewhat? Because we need to flatten a multi-agent env to be a single-agent env (as gym requires).

    NOTE: Observations returned are in a dictionary format with various information that is
     necessary to be able to handle the multi-agent nature of the environment. There are probably
     better ways to handle this, but we found this to work with minor modifications to OpenAI Baselines.

    NOTE: The index of the main agent (as gym envs are 'single-agent') in the mdp is randomized at each reset
     of the environment, and is kept track of by the self.agent_idx attribute. This means that it is necessary
     to pass on this information in the output to know for which agent index featurizations should be made for
     other agents.

    For example, say one is training A0 paired with A1, and A1 takes a custom state featurization.
    Then in the runner.py loop in OpenAI Baselines, we will get the lossless encodings of the state,
    and the true Overcooked state. When we encode the true state to feed to A1, we also need to know
    what agent index it has in the environment (as encodings will be index dependent).
    """

    env_name = "Overcooked-v0"

    # gym checks for the action space and obs space while initializing the env and throws an error if none exists
    # custom_init after __init__ no longer works
    # might as well move all the initilization into the actual __init__
    def __init__(self, base_env, featurize_fn, baselines_reproducible=False):
        """
        base_env: OvercookedEnv
        featurize_fn(mdp, state): fn used to featurize states returned in the 'both_agent_obs' field

        Example creating a gym env:

        mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
        base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
        env = gym.make("Overcooked-v0",base_env = base_env, featurize_fn =base_env.featurize_state_mdp)
        """
        if baselines_reproducible:
            # NOTE:
            # This will cause all agent indices to be chosen in sync across simulation
            # envs (for each update, all envs will have index 0 or index 1).
            # This is to prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which
            # seeding does not reach) i.e. having different results for different
            # runs with the same seed.
            # The effect of this should be negligible, as all other randomness is
            # controlled by the actual run seeds
            np.random.seed(0)
        self.base_env = base_env
        self.featurize_fn = featurize_fn
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.reset()
        self.visualizer = StateVisualizer()
        self.name = 'overcook'
        self.layout_name = self.base_env.layout_name

    def _setup_observation_space(self):
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.zeros(obs_shape)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        assert all(
            self.action_space.contains(a) for a in action
        ), "%r (%s) invalid" % (
            action,
            type(action),
        )
        agent_action, other_agent_action = [
            Action.INDEX_TO_ACTION[a] for a in action
        ]
        
        if self.agent_idx == 0:

            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, env_info = self.base_env.step(joint_action)
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)

        env_info["policy_agent_idx"] = self.agent_idx

        if "episode" in env_info.keys():
            env_info["episode"]["policy_agent_idx"] = self.agent_idx

        obs = {
            "both_agent_obs": both_agents_ob,
            "overcooked_state": next_state,
            "other_agent_env_idx": 1 - self.agent_idx,
        }
        return obs, reward, done, env_info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        self.mdp = self.base_env.mdp
        # self.agent_idx = np.random.choice([0, 1])
        self.agent_idx = 0
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        return {
            "both_agent_obs": both_agents_ob,
            "overcooked_state": self.base_env.state,
            "other_agent_env_idx": 1 - self.agent_idx,
        }

    def render(self):
        rewards_dict = {}  # dictionary of details you want rendered in the UI
        for key, value in self.base_env.game_stats.items():
            if key in [
                "cumulative_shaped_rewards_by_agent",
                "cumulative_sparse_rewards_by_agent",
            ]:
                rewards_dict[key] = value

        image = self.visualizer.render_state(
            state=self.base_env.state,
            grid=self.base_env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(
                self.base_env.state, **rewards_dict
            ),
        )

        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)
        image = cv2.resize(image, (2 * 528, 2 * 464))
        return image

    
    