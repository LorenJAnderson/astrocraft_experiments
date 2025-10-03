from .player_interface import PlayerInterface

import numpy as np

def normalize(array):
    return array / np.sum(array)

class OfflineDataGenerator(PlayerInterface):
    def __init__(self, n_agents, p_capture_slow=1, p_return_slow=1, p_capture_fast=1, p_return_fast=1, p_intercept_slow=1, p_intercept_fast=1, p_orbitals=[.5,0,0,0,0,0,.5], p_dodge=1, p_random_traj_change=0):
        self.first_turn = True
        self.n_agents = n_agents

        # Record prespecified probabilities for each action type
        self.p_orbitals = np.array(p_orbitals)
        self.p_capture_slow = p_capture_slow
        self.p_capture_fast = p_capture_fast
        self.p_return_slow = p_return_slow
        self.p_return_fast = p_return_fast
        self.p_intercept_slow = p_intercept_slow
        self.p_intercept_fast = p_intercept_fast
        self.p_dodge = p_dodge
        self.p_random_traj_change = p_random_traj_change

    def get_agent_status(self, state, agent):
        """Find out certain things about the agent such as:
            -Does it have the flag?
            -Is it in danger of being intercepted?
        """
        flagged = bool(state[agent+1,4])    # Check if agent is flagged

        # Check if any opponent nearby is transferring and close to the agent
        in_danger = any(np.logical_and(state[self.n_agents+2:,6] == 1, ((state[self.n_agents+2:,-1] - state[agent+1,-1]) % (2*np.pi) <= np.pi/12)))

        return flagged, in_danger


    def select_action(self, state, rew, term, trunc, info):
        """Selects an action according to the weighted probability distribution of the mask"""
        masks = state['action_mask']
        state = state['observation']

        # On turn 1, send agents to random orbitals
        if self.first_turn:
            self.first_turn = False
            P = normalize(self.p_orbitals*np.array([1,1,1,0,1,1,1]))
            actions = np.random.choice([1,2,3,4,5,6,7], size=self.n_agents, p=P)

        else:
            actions = np.zeros(self.n_agents)
            for agent in range(self.n_agents):
                    
                mask = masks[agent]

                try:
                    # If agent is mid transfer, do nothing
                    if sum(mask) == 1:
                        actions[agent] = 0
                        continue
                    
                except:
                    raise Exception(f"{mask}")

                # if agent is on orbital 0, transfer somewhere
                if state[agent+1,-2] == 0:
                    P = normalize(self.p_orbitals*np.array([1,1,1,0,1,1,1]))
                    actions[agent] = np.random.choice([1,2,3,4,5,6,7], p=P)
                    continue

                # Find which actions are available to it
                valid_actions = np.nonzero(mask)[0]
                flagged, in_danger = self.get_agent_status(state, agent)

                p = np.zeros_like(mask, dtype=np.float32)
                p[0] += .1

                # Add in probability of changing orbital for no particular reason
                p[1:8] += self.p_random_traj_change * mask[1:8] * self.p_orbitals # Chance of going to a particular orbital

                # If in danger, take dodge with p_dodge, select orbital with p_orbital
                if in_danger:
                    p[0] += 1-self.p_dodge  # Chance the agent chooses not to dodge
                    
                    p[1:8] += self.p_dodge * mask[1:8] * self.p_orbitals # Chance of going to a particular orbital

                # If flagged and able to return flag, return flag with probability p_return_fast/slow
                if flagged and mask[8] == 1:
                    p[8] += self.p_return_slow

                if flagged and mask[9] == 1:
                    p[9] += self.p_return_fast

                # If not flagged but able to capture flag, do so with probability p_capture_fast/slow
                if not flagged and mask[10] == 1:
                    p[10] += self.p_capture_slow

                if not flagged and mask[11] == 1:
                    p[11] += self.p_capture_fast

                # Fast intercept available targets with probability p_intercept_fast
                fasts = [x for x in valid_actions if x > 11 and x%2 == 1]
                slows = [x for x in valid_actions if x > 10 and x%2 == 0]

                for fast in fasts:
                    p[fast] += self.p_intercept_fast

                for slow in slows:
                    p[slow] += self.p_intercept_slow

                # Normalize P and select an action from P

                p = normalize(p)
                actions[agent] = np.random.choice([x for x in range(len(mask))], p=p)


        return actions
     

    def reset(self):
        self.first_turn = True
        return
