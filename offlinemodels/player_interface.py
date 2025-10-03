

class PlayerInterface():
    """An interface to be used in the creation of different players for the league. This
    will allow for agents of different types to be registered"""

    def __init__(self):
        return
    
    def select_action(self, state, rew, term, trunc, info):
        """Takes in the information received from the environment"""
        return
    
    def reset(self):
        """Used to signal to the agent that the game has been reset to its starting state"""
        return