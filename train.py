import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm.auto import tqdm
import os
from sys import argv

from offlinemodels.offline_generator import OfflineDataGenerator
from AstroCraft.PettingZoo_MA.env.CaptureTheFlagMA import CTFENVMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def normalize_state(state):
    """Normalizes state input"""
    new_state = state.copy()
    new_state[:,3] /= 1000  # Scale fuel to a percentage of max
    new_state[:,-1] /= 2*np.pi  # Scale angle to percentage of max
    new_state[:,-2] /= 3    # Scale orbital to (-1,1)

    return new_state

def gen_q_vals(r, gamma, n):
    """
    Generator for q-values given a reward r and discount factor gamma
    """
    for i in range(0,n):
        yield r*gamma**(n-i-1)

# Running averages for normalization
ema_f = 1.0
ema_g = 1.0
beta = 0.99  # smoothing factor

def normalized_objective(f, g, alpha):
    global ema_f, ema_g
    ema_f = beta * ema_f + (1-beta) * f.detach().abs().mean().item()
    ema_g = beta * ema_g + (1-beta) * g.detach().abs().mean().item()
    
    f_norm = f / (ema_f)
    g_norm = g / (ema_g)
    return f_norm + alpha * g_norm


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        # Create LSTM layer
        self.lstm = nn.LSTM(28, 128, batch_first=True)
        self.relu = nn.ReLU()
        
        # Create critic neural network layers
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 14)
        
        # Apply Xavier uniform initialization
        self._initialize_weights()
        
        self.net = nn.Sequential(self.relu, self.hidden1, self.relu, self.hidden2, self.relu, self.hidden3, self.relu, self.output)

    def _initialize_weights(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize fully connected layer weights
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.xavier_normal_(self.hidden2.weight)
        nn.init.xavier_normal_(self.hidden3.weight)
        nn.init.xavier_normal_(self.output.weight)
        
    def forward(self, state, lstm_hidden):
        # LSTM layer forward pass
        lstm_out, lstm_hidden = self.lstm(state, lstm_hidden)
        
        # Pass the state through the critic network
        value = self.net(lstm_out)
        
        return value, lstm_hidden

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        # Create LSTM layer
        self.lstm = nn.LSTM(28, 128, batch_first=True)
        self.relu = nn.ReLU()
        
        # Create critic neural network layers
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 14)
        
        # Apply Xavier uniform initialization
        self._initialize_weights()
        
        self.net = nn.Sequential(self.relu, self.hidden1, self.relu, self.hidden2, self.relu, self.hidden3, self.relu, self.output)

    def _initialize_weights(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize fully connected layer weights
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.xavier_normal_(self.hidden2.weight)
        nn.init.xavier_normal_(self.hidden3.weight)
        nn.init.xavier_normal_(self.output.weight)
        
    def forward(self, state, lstm_hidden):
        # LSTM layer forward pass
        lstm_out, lstm_hidden = self.lstm(state, lstm_hidden)
        
        # Pass the state through the critic network
        pol = self.net(lstm_out)
        
        return pol, lstm_hidden

class CQLA2C(nn.Module):
    """An A2C implementation with q-value pessimism built in"""

    def __init__(self, train_data):
        """Defines the neural network to be used"""
        super(CQLA2C, self).__init__()
        
        # Initialize policy and value heads
        self.value_net = ValueNet().to(device, non_blocking=True)
        self.policy_net = PolicyNet().to(device, non_blocking=True)

        # Initialize target policy and value heads
        self.value_net_t = ValueNet().to(device, non_blocking=True)
        self.value_net_t.load_state_dict(self.value_net.state_dict())

        self.train_data = train_data

        data_0_0 = {'states': [], 'actions':[], 'masks': [], 'rewards': [], 'done': []}
        data_1_0 = {'states': [], 'actions':[], 'masks': [], 'rewards': [], 'done': []}
        self.data = {0:{0: data_0_0}, 1:{0: data_1_0}}

        self.outcomes = []

    def forward(self, x, lstm_value, lstm_policy):
        """Conducts a forward pass of x through the network"""
        v, lstm_value = self.value_net(x, lstm_value)
        p, lstm_policy = self.policy_net(x, lstm_policy)

        return v,p,lstm_value,lstm_policy

    def value_objective_CQL(self, s, a, ns, r, d, m):
        """
        Calculates value loss for an episode
        """
        batch_size = s.size(0)
        (h0, c0) = (torch.zeros(1, batch_size, 128).detach().to(device), torch.zeros(1, batch_size, 128).detach().to(device))

        # Predict q-values for state-action pairs
        q_all, _ = self.value_net.forward(s, (h0, c0))
        q_preds = q_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            q_all_t, _ = self.value_net_t(s, (h0, c0))
            q_all_t_next = torch.roll(q_all_t, shifts=-1, dims=1)
            q_all_t_next[:, -1, :] = 0.0
            
            q_all_t_next.masked_fill(~m, -float("inf"))
            
            q_next = q_all_t_next.max(dim=-1).values

        # Compute Bellman error
        bellman_target = r + q_next * (1 - d.float())
        bellman = 0.5 * F.huber_loss(q_preds, bellman_target.detach(), reduction='mean')
        
        # Compute CQLH regularization term
        q_all = q_all.masked_fill(~m, -float("inf"))
        
        lse = torch.logsumexp(q_all, dim=-1)
        cqlh = self.alpha * (lse-q_preds).mean()
        loss = normalized_objective(bellman, cqlh, self.alpha)
        return loss
    
    def value_objective_Online(self, s, a, ns, r, d, m):
        """
        Calculates value loss for an episode
        """
        batch_size = s.size(0)
        (h0, c0) = (torch.zeros(1, batch_size, 128).detach().to(device), torch.zeros(1, batch_size, 128).detach().to(device))

        # Predict q-values for state-action pairs
        q_all, _ = self.value_net.forward(s, (h0, c0))
        q_preds = q_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            q_all_t, _ = self.value_net_t(s, (h0, c0))
            q_all_t_next = torch.roll(q_all_t, shifts=-1, dims=1)
            q_all_t_next[:, -1, :] = 0.0
            
            q_all_t_next.masked_fill(~m, -float("inf"))
            
            q_next = q_all_t_next.max(dim=-1).values

        # Compute Bellman error
        bellman_target = r + q_next * (1 - d.float())
        bellman = 0.5 * F.huber_loss(q_preds, bellman_target.detach(), reduction='mean')

        return bellman 
    
    def policy_objective(self, s, m):
        """
        Calculates the policy loss for an episode, including the entropy bonus.
        """
        batch_size = s.size(0)
        (h0, c0) = (torch.zeros(1, batch_size, 128).detach().to(device), torch.zeros(1, batch_size, 128).detach().to(device))

        # Get Q-value predictions from the target value network
        q_preds, _ = self.value_net_t.forward(s, (h0, c0))
        q_preds = q_preds.detach()

        # Get action probabilities from the policy network
        logits, _ = self.policy_net.forward(s, (h0, c0))
        # Apply masking and softmax to get action probabilities
        logits = logits.masked_fill(~m, -float("inf"))
        probs = F.softmax(logits, dim=-1)
        # print(probs)
        # raise Exception
        
        # Optional safety: make sure each state has at least one valid action
        if not m.any(dim=-1).all():
            raise ValueError("Found a state with no valid actions.")

        # Calculate the entropy
        logp = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * logp).sum(dim=-1)
        
        # Advantage
        v = (probs * q_preds).sum(dim=-1, keepdim=True).detach()
        adv = (q_preds - v).masked_fill(~m, 0.0)
        expected_adv = (probs * adv).sum(dim=-1)

        # Calculate the policy objective
        #loss = -normalized_objective(expected_adv, entropy, self.beta).mean()
        loss = -(expected_adv + self.beta * entropy).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite")
        return loss

    def fit(self, env, val_optim, pol_optim, burn_in=1, fine_tune=1, tau=.05, alpha=1e-3, beta_offline=1e-3, beta_online=1e-3, batch_size_offline=32, batch_size_online=32, weights_path=None):
        """Trains the neural network on the input dataset for the specified amount of time"""
        self.v_losses = []
        self.p_losses = []
        self.outcomes = []
        self.alpha = alpha
        
        # Burn in using offline data
        train_loader = DataLoader(self.train_data, batch_size=1, shuffle=True)
        self.beta = beta_offline
        
        for epoch in range(burn_in):
            loop = tqdm(total=len(train_loader), position=0, leave=False, ncols=50)
            for batch, trajectory in enumerate(train_loader):
                
                s, a, ns, m, r, d = trajectory['states'].to(device, non_blocking=True), trajectory['actions'].to(device, non_blocking=True), trajectory['next_states'].to(device, non_blocking=True), trajectory['masks'].to(device, non_blocking=True), trajectory['rewards'].to(device, non_blocking=True), trajectory['done'].to(device, non_blocking=True)  
                # Get loss for actor and critic
                val_loss = self.value_objective_CQL(s, a, ns, r, d, m)
                policy_loss = self.policy_objective(s, m)
                self.v_losses.append(val_loss.item())
                self.p_losses.append(policy_loss.item())

                # Backprop and step
                val_loss.backward()
                policy_loss.backward()

                if batch % batch_size_offline == 0 and batch > 0:
                    #val_loss = clip_grad_norm_(self.value_net.parameters(), max_norm=5.0)
                    val_optim.step()
                    val_optim.zero_grad()
                    #policy_loss = clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                    pol_optim.step()
                    pol_optim.zero_grad()
                    loop.set_description(f"ep:{str(epoch)}\tpol:{np.mean(self.p_losses[-batch_size_offline:])}\tval:{np.mean(self.v_losses[-batch_size_offline:]): .5f}")
                    loop.update(batch_size_offline)

                    # update target networks 
                    for target_param, param in zip(self.value_net_t.parameters(), self.value_net.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
            loop.close()
            
            # Save weights every few epochs
            if weights_path is not None and epoch % (burn_in // 10) == 0:
                self.save_weights(weights_path + "/_offline_{}.pth".format(epoch))

        # Fine tune using online data in environment
        most_wins = self.outcomes[-100:].count(1) + .2 * self.outcomes[-100:].count(0)
        best_model = 0
        loop = tqdm(total=fine_tune, position=0, leave=False, ncols=50)
        self.beta_online = beta_online
        for episode in range(fine_tune):
            
            team = np.random.randint(2)
            if team == 0:
                data = self.play_game_blue(env)
            else:
                data = self.play_game_red(env)
            
            s = data[team][0]['states']
            ns = s[1:]
            ns.append(s[-1])
            a = data[team][0]['actions']
            m = data[team][0]['masks'][1:] + [data[team][0]['masks'][-1]]
            r = data[team][0]['rewards']
            d = data[team][0]['done']
            
            s = [x.flatten() for x in s]
            ns = [x.flatten() for x in ns]
            a = [x for x in a]
            m = [x.T for x in m]
            r = [x for x in r]
            d = [x for x in d]

            s, a, ns, m, r, d = torch.FloatTensor(s).to(device, non_blocking=True), torch.LongTensor(a).to(device, non_blocking=True), torch.FloatTensor(ns).to(device, non_blocking=True), torch.BoolTensor(m).to(device, non_blocking=True), torch.FloatTensor(r).to(device, non_blocking=True), torch.BoolTensor(d).to(device, non_blocking=True)  
            
            # Get loss for actor and critic
            val_loss = self.value_objective_Online(s.unsqueeze(0), a.unsqueeze(0), ns.unsqueeze(0), r.unsqueeze(0), d.unsqueeze(0), m.unsqueeze(0))
            policy_loss = self.policy_objective(s.unsqueeze(0), m)
            self.v_losses.append(val_loss.item())
            self.p_losses.append(policy_loss.item())

            # Backprop and step
            val_loss.backward()
            policy_loss.backward()

            if episode % batch_size_online == 0:
                val_optim.step()
                val_optim.zero_grad()
                pol_optim.step()
                pol_optim.zero_grad()
                loop.set_description(f"ep:{str(episode)}\tW\L\T:{(len([x for x in self.outcomes[-min(len(self.outcomes), 100):] if x > 0]), len([x for x in self.outcomes[-min(len(self.outcomes), 100):] if x < 0]), len([x for x in self.outcomes[-min(len(self.outcomes), 100):] if x == 0]))}\tpol:{np.mean(self.p_losses[-batch_size_online:]): .5f}\tval:{np.mean(self.v_losses[-batch_size_online:]): .5f}")
                loop.update(batch_size_online)

                # update target networks 
                for target_param, param in zip(self.value_net_t.parameters(), self.value_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            # Save weights if model is the best so far
            if self.outcomes[-100:].count(1) + .2 * self.outcomes[-100:].count(0) > most_wins:
                most_wins = self.outcomes[-100:].count(1) + .1 * self.outcomes[-100:].count(0)
                self.save_weights(weights_path + "/_best_{}.pth".format(episode))

            # Save weights every few epochs
            if episode % 100 == 0 and weights_path is not None:
                self.save_weights(weights_path + "/_online_{}.pth".format(episode // 100))
                
        loop.close()
        
    def select_action(self, state, rew, term, trunc, info, hn, cn):
        """Selects a set of actions based on the current state and available actions"""
        mask = torch.BoolTensor(state['action_mask'][0]).to(device, non_blocking=True)
        state = state['observation']

        with torch.no_grad():
            state = torch.from_numpy(normalize_state(state)[:,2:].flatten()).unsqueeze(0)
            state = state.to(device)
            logits,(hn,cn) = self.policy_net.forward(state, (hn, cn))
            logits = logits.masked_fill(~mask, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            return np.array([torch.multinomial(probs, 1).item()]), (hn, cn)
        
    def play_game_blue(self, env, epsilon=0):
        # Reset environment
        state, info = env.reset()
        rew = {'player0': 0, 'player1': 0}
        term = {'player0': False, 'player1': False}
        trunc = {'player0': False, 'player1': False}

        # Build a bot
        p_capture_slow = np.random.uniform(0,.5)
        p_return_slow = np.random.uniform(.5,.7)
        p_capture_fast = np.random.uniform(.337,1)
        p_return_fast = np.random.uniform(.45,1)
        p_intercept_slow = np.random.uniform(.5,1)
        p_intercept_fast = np.random.uniform(.62,1)
        orb_norm = 0
        while abs(orb_norm - 1) > .1:
            p_orbital_1 = np.random.uniform(.1,.2)
            p_orbital_2 = np.random.uniform(.17,.2)
            p_orbital_3 = np.random.uniform(.12,.15)
            p_orbital_4 = np.random.uniform(0,.12)
            p_orbital_5 = np.random.uniform(0,.12)
            p_orbital_6 = np.random.uniform(.18,.27)
            p_orbital_7 = np.random.uniform(0,.17)
            orbitals = [p_orbital_1, p_orbital_2, p_orbital_3, p_orbital_4, p_orbital_5, p_orbital_6, p_orbital_7]
            orb_norm = sum(orbitals)

        orbitals = [x/orb_norm for x in orbitals]

        p_dodge = np.random.uniform(.48,1)
        p_random_traj_change = 0

        bot = OfflineDataGenerator(1, p_capture_slow, p_return_slow, p_capture_fast, p_return_fast, p_intercept_slow, p_intercept_fast, orbitals, p_dodge, p_random_traj_change)

        # Assign model to blue team and bot to red
        team = 0
            
        (hn, cn) = (torch.zeros(1,128).detach().to(device), torch.zeros(1,128).detach().to(device))
        first_turn = True
        
        # Continue playing until the game is over
        while True:
            action0, (hn,cn) = self.select_action(state['player0'], rew['player0'], term['player0'], trunc['player0'], info['player0'], hn, cn)
            action1 = bot.select_action(state['player1'], rew['player1'], term['player1'], trunc['player1'], info['player1'])

            # Check if action is to be randomly selected
            eps = np.random.rand()
            if eps < epsilon:
                if team == 0:
                    mask = state['player0']['action_mask']
                    avail_act = np.nonzero(mask[0])[0]
                    action0 = np.array([np.random.choice(avail_act)])
                else:
                    mask = state['player1']['action_mask']
                    avail_act = np.nonzero(mask[0])[0]
                    action1 = np.array([np.random.choice(avail_act)])
                    

            action = {'player0': action0, 'player1': action1}
            state0 = state['player0']
            state1 = state['player1']
            state, rew, term, trunc, info = env.step(action)
            rew0 = rew['player0']
            rew1 = rew['player1']
            self.add_data(normalize_state(state0['observation'])[:,2:], normalize_state(state1['observation'])[:,2:], action0, action1, rew0, rew1, state0['action_mask'], state1['action_mask'], term['player0'] or term['player1'], team)

            if term['player0'] or term['player1'] or (trunc['player0'] and trunc['player1']):
                if team == 0 and hasattr(self, "outcomes"):
                    self.outcomes.append(rew0)
                elif hasattr(self, "outcomes"):
                    self.outcomes.append(rew1)
                break

        return self.save_batch()

    def play_game_red(self, env, epsilon=0):
        # Reset environment
        state, info = env.reset()
        rew = {'player0': 0, 'player1': 0}
        term = {'player0': False, 'player1': False}
        trunc = {'player0': False, 'player1': False}

        # Build a bot
        p_capture_slow = np.random.uniform(0,.5)
        p_return_slow = np.random.uniform(.5,.7)
        p_capture_fast = np.random.uniform(.337,1)
        p_return_fast = np.random.uniform(.45,1)
        p_intercept_slow = np.random.uniform(.5,1)
        p_intercept_fast = np.random.uniform(.62,1)
        orb_norm = 0
        while abs(orb_norm - 1) > .1:
            p_orbital_1 = np.random.uniform(.1,.2)
            p_orbital_2 = np.random.uniform(.17,.2)
            p_orbital_3 = np.random.uniform(.12,.15)
            p_orbital_4 = np.random.uniform(0,.12)
            p_orbital_5 = np.random.uniform(0,.12)
            p_orbital_6 = np.random.uniform(.18,.27)
            p_orbital_7 = np.random.uniform(0,.17)
            orbitals = [p_orbital_1, p_orbital_2, p_orbital_3, p_orbital_4, p_orbital_5, p_orbital_6, p_orbital_7]
            orb_norm = sum(orbitals)

        orbitals = [x/orb_norm for x in orbitals]

        p_dodge = np.random.uniform(.48,1)
        p_random_traj_change = 0

        bot = OfflineDataGenerator(1, p_capture_slow, p_return_slow, p_capture_fast, p_return_fast, p_intercept_slow, p_intercept_fast, orbitals, p_dodge, p_random_traj_change)

        # Assign the model to the red team and a bot to blue
        team = 1
            
        (hn, cn) = (torch.zeros(1,128).detach().to(device), torch.zeros(1,128).detach().to(device))
        
        first_turn = True
        
        # Continue playing until the game is over
        while True:
            action1, (hn,cn) = self.select_action(state['player1'], rew['player1'], term['player1'], trunc['player1'], info['player1'], hn, cn)
            action0 = bot.select_action(state['player0'], rew['player0'], term['player0'], trunc['player0'], info['player0'])
            # Check if action is to be randomly selected
            eps = np.random.rand()
            if eps < epsilon:
                if team == 0:
                    mask = state['player0']['action_mask']
                    avail_act = np.nonzero(mask[0])[0]
                    action0 = np.array([np.random.choice(avail_act)])
                else:
                    mask = state['player1']['action_mask']
                    avail_act = np.nonzero(mask[0])[0]
                    action1 = np.array([np.random.choice(avail_act)])

            action = {'player0': action0, 'player1': action1}
            state0 = state['player0']
            state1 = state['player1']
            state, rew, term, trunc, info = env.step(action)
            rew0 = rew['player0']
            rew1 = rew['player1']
            
            if first_turn:
                first_turn = False
            
            self.add_data(normalize_state(state0['observation'])[:,2:], normalize_state(state1['observation'])[:,2:], action0, action1, rew0, rew1, state0['action_mask'], state1['action_mask'], term['player0'] or term['player1'], team)

            if term['player0'] or term['player1'] or (trunc['player0'] and trunc['player1']):
                if team == 0 and hasattr(self, "outcomes"):
                    self.outcomes.append(rew0)
                elif hasattr(self, "outcomes"):
                    self.outcomes.append(rew1)
                break

        return self.save_batch()
        
    def save_batch(self):
        """Saves current data from batch"""
        data_0_0 = {'states': [], 'actions':[], 'masks': [], 'rewards': [], 'done': []}
        data_1_0 = {'states': [], 'actions':[], 'masks': [], 'rewards': [], 'done': []}
        data = self.data
        self.data = {0:{0: data_0_0}, 1:{0: data_1_0}}
        return data

    def add_data(self, state0, state1, action0, action1, rew0, rew1, mask0, mask1, done, team):
        """Appends transition information to proper attributes. Saves and clears each cache if necessary."""
        states = [state0, state1]
        actions = [action0, action1]
        rewards = [rew0, rew1]
        masks = [mask0, mask1]
        trivial = sum(masks[team][0]) <= 1

        # Only record data if a non-trivial action was taken by the agent, or if the game is over.
        if not trivial and not done: 
            #print(f"Non trivial transition\tAppending {rewards[team]}")
            self.data[team][0]['states'].append(states[team])
            self.data[team][0]['actions'].append(actions[team][0])
            self.data[team][0]['masks'].append(masks[team][0])
            self.data[team][0]['rewards'].append(rewards[team])
            self.data[team][0]['done'].append(done)

        elif rewards[team] != 0:
            #print(f"Game finished\tAppending {rewards[team]}")
            self.data[team][0]['rewards'][-1] = rewards[team]
            self.data[team][0]['done'][-1] = True

    def save_weights(self, weights_path):
        """Saves the model's weights at the specified path"""
        torch.save(self.state_dict(), weights_path)

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, weights_only=False, map_location=device))

class AstroCraftData(Dataset):
    """This dataset is comprised of recorded 3v3 astrocraft games"""

    def __init__(self, folder, rew_transform=None):

        if rew_transform == None:
            self.rew_transform = np.vectorize(lambda x: x)
        else:
            self.rew_transform = np.vectorize(rew_transform)

        self.sequences = []

        # Open each stored episode and group the data by timestep
        with torch.no_grad():
            for root, dirs, files in os.walk(folder, topdown=False):
                loop = tqdm(total=len(files), position=0, leave=False)
                for item in files:
                    name = os.path.join(root, item)
                    if os.path.isfile(name):
                        loop.set_description(f"Reading {name}...")
                        loop.update()

                        data = np.load(name, allow_pickle=True)

                        for team in [0,1]:
                            for agent in [0]:
                                s = data[team][agent]['states']
                                ns = s[1:]
                                ns.append(s[-1])
                                a = data[team][agent]['actions']
                                m = data[team][agent]['masks']
                                r = data[team][agent]['rewards']
                                d = data[team][agent]['done']
                                
                                s = [normalize_state(x)[:,2:].flatten() for x in s]
                                ns = [normalize_state(x)[:,2:].flatten() for x in ns]
                                a = [x for x in a]
                                m = [x.T for x in m]
                                r = [x for x in r]
                                d = [x for x in d]
                                
                                s = torch.FloatTensor(s)
                                a = torch.LongTensor(a)
                                ns = torch.FloatTensor(ns)
                                r = torch.FloatTensor(r)
                                m = torch.BoolTensor(m)
                                d = torch.BoolTensor(d)

                                self.sequences.append({'states':torch.FloatTensor(s), 'actions':torch.LongTensor(a), 'next_states':torch.FloatTensor(ns), 'masks':torch.BoolTensor(m), 'rewards':torch.FloatTensor(r), 'done':torch.BoolTensor(d)})
                
                loop.close()

    def extend(self, data, replace=True):
        """Takes in a new set of transitions and adds them to the dataset. If replace=True,
        an equivalent number of old transitions are removed first."""
        
        # Remove an equivalent number of old transitions
        if replace:
            n = max(len(data[0][0]['actions']), len(data[1][0]['actions']))
            self.sequences = self.sequences[n:] #14872

        # Add new transitions
        with torch.no_grad():
            for team in [0,1]:
                for agent in [0]:
                    s = data[team][agent]['states']
                    if len(s) > 0:
                        ns = s[1:]
                        ns.append(s[-1])
                        a = data[team][agent]['actions']
                        m = data[team][agent]['masks']
                        r = data[team][agent]['rewards']
                        d = data[team][agent]['done']

                        s = [torch.from_numpy(normalize_state(x)[:,2:].flatten()).float() for x in s]
                        ns = [torch.from_numpy(normalize_state(x)[:,2:].flatten()).float() for x in ns]
                        a = [torch.tensor(x).long() for x in a]
                        m = [torch.from_numpy(x.T).bool() for x in m]
                        r = [torch.tensor(x).float() for x in r]
                        d = [torch.tensor(x).bool() for x in d]

                        self.sequences.append({'states':torch.FloatTensor(s), 'actions':torch.LongTensor(a), 'next_states':torch.FloatTensor(ns), 'masks':torch.BoolTensor(m), 'rewards':torch.FloatTensor(r), 'done':torch.BoolTensor(d)})

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Return the (s, a, q) pair at idx"""
        return self.sequences[idx]

    def rew_to_q(self, rewards):
        q_values = np.zeros_like(rewards, dtype=float)
    
        for t in range(len(rewards) - 1, -1, -1):
            if t == len(rewards) - 1:
                # Q-value for the last time step is simply the last reward
                q_values[t] = rewards[t]
            else:
                # Q-value at time t is the sum of the current reward and the discounted
                # Q-value of the next time step
                q_values[t] = rewards[t] + self.gamma * q_values[t + 1]

        return q_values

if __name__ == "__main__":
    # IMPORTANT!
    # Be sure to pass in a filepath to save the weights when running this file!
    
    dataset = AstroCraftData("./offline_data/")    # Change this to wherever your offline data is stored (will search recursively through subfolders)

    model = CQLA2C(dataset)
    val_optim = optim.AdamW(model.value_net.parameters(), 1e-4, amsgrad=True)
    pol_optim = optim.AdamW(model.policy_net.parameters(), 1e-4, amsgrad=True)
    env = CTFENVMA(1, 1, 0)

    try:
        model.fit(env, val_optim, pol_optim, burn_in=1000, fine_tune=10000, tau=.005, alpha=1e-2, beta_offline=1e-3, beta_online=1, batch_size_offline=32, batch_size_online=1, weights_path=argv[1])
        model.save_weights(argv[1] + "/_final.pth")
    except KeyboardInterrupt as e:
        model.save_weights(argv[1] + "/_final.pth")
        raise e
    