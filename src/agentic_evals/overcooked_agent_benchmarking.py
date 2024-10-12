import os
import sys
# Get the parent directory (e.g. for overcooked_ai_py)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from llm_coordination_agents.overcooked_action_manager import LLMActionManager
from llm_coordination_agents.scripted_action_manager import ScriptedActionManager
from overcooked_ai_py.mdp.actions import Action, Direction
import time 
import numpy as np 
from tqdm import tqdm 
import argparse

import pdb
#from vllm import LLMEngine
#
#class LMMEngineAzureOpenAI(LMMEngine):
#    def __init__(self, api_key=None, azure_endpoint=None, model=None, api_version=None, rate_limit=-1, **kwargs):
#        assert model is not None, "model must be provided"
#        self.model = model
#        assert api_version is not None, "api_version must be provided"
#        self.api_version = api_version
#        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
#        if api_key is None:
#            raise ValueError("An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY")
#        self.api_key = api_key
#        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_API_BASE")
#        if azure_endpoint is None:
#            raise ValueError("An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_API_BASE")
#        self.azure_endpoint = azure_endpoint
#        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
#        self.llm_client = AzureOpenAI(azure_endpoint=self.azure_endpoint, api_key=self.api_key, api_version=self.api_version)
#        self.cost = 0.
#    # @backoff.on_exception(backoff.expo, (APIConnectionError, APIError, RateLimitError), max_tries=10)
#    def generate(self, messages, temperature=0., max_new_tokens=None, **kwargs):
#        '''Generate the next message based on previous messages'''
#        completion = self.llm_client.chat.completions.create(
#            model=self.model,
#            messages=messages,
#            max_tokens=max_new_tokens if max_new_tokens else 4096,
#            temperature=temperature,
#            **kwargs,
#        )
#        total_tokens = completion.usage.total_tokens
#        self.cost +=  0.02 * ((total_tokens+500) / 1000)
#        return completion.choices[0].message.content


def main(layout_name, model_name, scripted_collab, reflexion):
    mdp = OvercookedGridworld.from_layout_name(layout_name)

    if not scripted_collab:
        am = [LLMActionManager(mdp, 'player_0', layout_name, model_name),
              LLMActionManager(mdp, 'player_1', layout_name, model_name)]
    else:
        am = [ScriptedActionManager(mdp, 'player_0', layout_name, 'do_nothing'),
              ScriptedActionManager(mdp, 'player_1', layout_name, scripted_collab)]
       
        
        
    state = mdp.get_standard_start_state()
    #print(am[0].llm_agent.model)
    # print(action)
    # game_messages[action_manager.player_id] = message 
    score = 0
    NUM_TICKS = 400
    for tick in tqdm(range(NUM_TICKS)):
        joint_action = [Action.STAY] * 2

        for i in range(2):
            print("current i: ", i)
            #pdb.set_trace()
            action, message = am[i].get_next_move(state, '')
            joint_action[i] = action 
        # print(joint_action)
        # Apply overcooked game logic to get state transition
        prev_state = state
        state, sparse_reward, shaped_reward = mdp.get_state_transition(
            prev_state, joint_action
        )
        info = {
            'sparse_reward_by_agent': sparse_reward, 
            'shaped_reward_by_agent': shaped_reward
        }
            
        # Update score based on soup deliveries that might have occured
        curr_reward = sparse_reward
        score += curr_reward
        if tick % 50 == 0:
            print(f"Current Score: {score}")
        print("Current Tick is: ", tick)
        print(mdp.state_string(state))
        print(f"Current score is : {score}")
        time.sleep(0.5) # Delay to avoid overloading LLM API with calls

    pdb.set_trace()
    if reflexion:
        reflexion_agent = LLMActionManager(mdp, 'player_0', layout_name, model_name, reflector=True)
        prev_game_results = '\nFinal score: ' + score
        prev_game_results += '\nMy action history: ' + am[0].llm_agent.self._add_history()
        prev_game_results += '\nMy teammate\s action history: ' + am[1].llm_agent.self._add_history()
        reflection = 'Reflection on previous match: ' + reflexion_agent.get_reflection(prev_game_results)
        
    return score

def _read_history(self):
    # Check if action history exists in the log
    if "action_history" in self.log_csv_dict:
        # Retrieve the action history from the log
        action_history = self.log_csv_dict["action_history"]
        # Format it as a string, joining the list items
        return f"Action history: {', '.join(action_history)}."
    else:
        return "No action history found."


parser = argparse.ArgumentParser(description='Run Overcooked benchmark with a specific model.')
parser.add_argument('model_name', default='', type=str, help='The name of the model to benchmark')
parser.add_argument('scripted_collab', default='none', choices=['none', 'onion_only', 'do_nothing'],
                    help='Designate the behavior model for teammate')
parser.add_argument('reflexion', default='false', choices=['false', 'true'])
args = parser.parse_args()

model_name = args.model_name
scripted_collab = args.scripted_collab
reflexion = args.reflexion
print(f'Benchmarking model: {model_name}')
print(f'Scripted partner: {scripted_collab}')


if __name__ == '__main__':
    LAYOUTS = ['cramped_room', 'forced_coordination', 'counter_circuit_o_1order', 'asymmetric_advantages', 'coordination_ring']
    NUM_TRIALS = 3
    
    for layout_name in LAYOUTS:
        scores = []
        gpt_3_costs = []
        gpt_4_costs = []
        for idx in range(NUM_TRIALS):
            score = main(layout_name, model_name, scripted_collab, reflexion)
            scores.append(score)

        with open(f'{layout_name}.txt', 'w') as f:
            f.write("MODEL: GPT4-turbo",)
            f.write(f"MEAN SCORE: {np.mean(scores)}\n")
            f.write(f"STD ERROR: {np.std(np.array(scores)) / np.sqrt(NUM_TRIALS)}\n")
            f.write(f"SAMPLE SCORES: {scores}\n")

    
