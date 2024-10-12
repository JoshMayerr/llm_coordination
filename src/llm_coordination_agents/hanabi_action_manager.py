from __future__ import print_function

import numpy as np
from hanabi_learning_environment import pyhanabi
from llm_coordination_agents.hanabi_agent import LLMAgent
import datetime 
import gym

def ai_score(fireworks):
    return np.sum(fireworks)

def run_game(game_parameters, model_name):
    game = pyhanabi.HanabiGame(game_parameters)
    print(game.parameter_string(), end="")
    state = game.new_initial_state()
    Players = [LLMAgent(0, model_name), LLMAgent(1, model_name)]
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = 'random'
    if Players[0].model_type != 'openai':
        model_name = 'Mixtral'
    else:
        model_name = Players[0].model
    game_name = f'TEST_HanabiGamePlay_{time_stamp}_score_model_{model_name}_seed_{game_parameters["seed"]}.txt'
    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue

        observation = state.observation(state.cur_player())
        episodic_memory = [Players[0].action_history, Players[1].action_history]
        working_memory = [Players[0].working_memory, Players[1].working_memory]
        # print_encoded_observations(obs_encoder, state, game.num_players())

        move = Players[state.cur_player()-1].get_next_move(observation, episodic_memory, working_memory)

        print("Selected Move: {}".format(move))

        state.apply_move(move)

    print("")
    print("Game done. Terminal state:")
    print("")
    print(state)
    print("")

    # This score is the total number of cards placed without considering bombing 
    print("Score: {}".format(ai_score(state.fireworks())))
    print("Bombed: {}".format('YES' if state.score() == 0 else 'NO'))


class HanabiEnvironment(gym.Env):
    def __init__(self, game_parameters):
        self.game_parameters = game_parameters
        self.game = pyhanabi.HanabiGame(game_parameters)
        self.state = self.game.new_initial_state()

        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.game_name = f'TEST_HanabiGamePlay_{self.time_stamp}_score_model_{self.model_name}_seed_{self.game_parameters["seed"]}.txt'
        
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(self.game.num_players(), 5), dtype=np.int32)

    def reset(self):
        self.state = self.game.new_initial_state()
        return self.state.observation(0)

    def step(self, action):
        if self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()
            return self.state.observation(0), 0, False, {}
        observation = self.state.observation(self.state.cur_player())
        episodic_memory = [self.Players[0].action_history, self.Players[1].action_history]
        working_memory = [self.Players[0].working_memory, self.Players[1].working_memory]
        move = self.Players[self.state.cur_player()-1].get_next_move(observation, episodic_memory, working_memory)
        self.state.apply_move(move)
        if self.state.is_terminal():
            return self.state.observation(0), ai_score(self.state.fireworks()), True, {}
        return self.state.observation(0), 0, False, {}

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass