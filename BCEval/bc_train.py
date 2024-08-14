import numpy as np
import torch
import torch.nn.functional as F
import os
from typing import List
from transformers import BertTokenizer, BertModel
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld, 
    PlayerState,
    Action, 
    Direction 
)

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import tqdm

import gym

from PIL import Image
import os
import time

from typing import List

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

STR2ACT = {
    'north': n,
    'south': s,
    'east': e,
    'west': w,
    'stay': stay,
    'interact': interact
}

ACTLIST = list(STR2ACT.keys())

device = 'cuda'
agent_list = ['Agent0', 'Agent1']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
extractor = BertModel.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, name):
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.name = name

    # 
    def learn(self, states, actions):
        probs = self.policy(states)
        actions=actions.squeeze(1)
        bc_loss = self.loss_fn(probs, actions)
        
        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def save(self, prefix):
        if not os.path.exists(prefix):
            os.makedirs(prefix, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(prefix, f'{self.name}_policy.pth'))
        
    def load(self, prefix):
        self.policy.load_state_dict(torch.load(os.path.join(prefix, f'{self.name}_policy.pth')))
  

def getActs(agent1, agent2, textObs):
    def textObs2vec(textObs_: List[str]):
        if isinstance(textObs_, str):
            textObs_ = [textObs_]
        res = []
        for t in textObs_:
            tokens_ = tokenizer.tokenize(t)
            tokens_ = ['[CLS]'] + tokens_ + ['[SEP]']

            indexed_tokens_ = tokenizer.convert_tokens_to_ids(tokens_)
            tokens_tensor_ = torch.tensor([indexed_tokens_])
            with torch.no_grad():
                outputs_ = extractor(tokens_tensor_)
                cur_v = outputs_.last_hidden_state[:, 0, :]
            res.append(cur_v)
            
        res = torch.stack(res, dim=0).view(len(textObs_), -1).detach().cpu().numpy()
        # print('in orig ', res.shape)
        # print(res.shape)
        # exit(0)
        return res

    obs = textObs2vec(textObs)
    agent1_act = agent1.take_action(obs)
    agent2_act = agent2.take_action(obs)
    return [agent1_act, agent2_act]
        
    
def bc_train(path_dir: str = "./overcooked_expert_data/", layout_names: List[str] = ['cramped_room', 'forced_coordination']):
    lr = 1e-3
    batch_size = 64
    state_dim = 768
    hidden_dim = 256
    act_dim = 6
    n_iterations = int(6.6e5)
    save_dir = './bc_models'
    
    for layout_name in layout_names:
        cur_dir = os.path.join(path_dir, layout_name)
        obs_str = torch.tensor(np.load(os.path.join(cur_dir, 'obs.npy')), dtype=torch.float32).view(-1, 768)
        agent1_act = torch.LongTensor(np.load(os.path.join(cur_dir, 'act1.npy'))).view(-1, 1)
        agent2_act = torch.LongTensor(np.load(os.path.join(cur_dir, 'act2.npy'))).view(-1, 1)
        cur_shapes = obs_str.shape[0]
        bc_agent1 = BehaviorClone(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=act_dim, lr=lr, name='agent1')
        bc_agent2 = BehaviorClone(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=act_dim, lr=lr, name='agent2')
        cur_dir = os.path.join(save_dir, layout_name)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir, exist_ok=True)
        with tqdm.tqdm(total=n_iterations, desc="n_iterations") as pbar:
            for i in range(n_iterations):
                sample_indices = np.random.randint(low=0, high=cur_shapes, size=batch_size)
                bc_agent1.learn(obs_str[sample_indices], agent1_act[sample_indices])
                bc_agent2.learn(obs_str[sample_indices], agent2_act[sample_indices])
                pbar.update(1)
        
        bc_agent1.save(prefix=cur_dir)
        bc_agent2.save(prefix=cur_dir)
       
        
def eval_bc(layout_name: str = "cramped_room"):
    
    ep_limit = {
        'cramped_room': 20, 
        'forced_coordination': 25
    }   
    state_dim = 768
    hidden_dim = 256
    act_dim = 6
    test_times = 10
    params = {
        "num_items_for_soup": 2,
        "rew_shaping_params": None,
        "cook_time": 2,
        "start_order_list": None
    } 
    bc_agent1 = BehaviorClone(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=act_dim, lr=0, name='agent1')
    bc_agent2 = BehaviorClone(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=act_dim, lr=0, name='agent2')
    
    bc_agent1.load(os.path.join('./bc_models',layout_name))
    bc_agent2.load(os.path.join('./bc_models',layout_name))
    time_str = time.strftime("%d%H%M%S", time.localtime())
    if not os.path.exists(os.path.join('./bc_results', layout_name, time_str)):
        os.makedirs(os.path.join('./bc_results', layout_name, time_str), exist_ok=True)
    total_success = 0
    for _ in range(test_times):
        base_mdp = OvercookedGridworld.from_layout_name(layout_name, **params)
        base_env = OvercookedEnv.from_mdp(base_mdp, horizon=100)
        gym_env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
        base_env.reset()
        gym_env.reset() 
        frames = []
        img = gym_env.render()
        frames.append(Image.fromarray(img))
        done = False
        step = 0
        while not done and step<ep_limit[layout_name]:
            text_obs = base_env.get_preety()
            cur_act = getActs(bc_agent1, bc_agent2, text_obs)
            obs, rew, done, info = gym_env.step(cur_act)
            img = gym_env.render()
            frames.append(Image.fromarray(img))
            done = base_env.get_cooked_one()
            step += 1
            
        if done:
            total_success += 1
        frames[0].save(f"./bc_results/{layout_name}/{time_str}/{_}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
        
    print(f'bc agent in {layout_name} SR: {total_success/test_times}')  

        
if __name__ == '__main__':

    layout_name = ["forced_coordination", 'cramped_room']
    bc_train(layout_names=layout_name)

    for layout_name in layout_name:
        eval_bc(layout_name)