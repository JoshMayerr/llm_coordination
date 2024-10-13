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


def main(layout_name, model_name, scripted_collab, reflections):
    mdp = OvercookedGridworld.from_layout_name(layout_name)

    if not scripted_collab:
        am = [LLMActionManager(mdp, 'player_0', layout_name, model_name),
              LLMActionManager(mdp, 'player_1', layout_name, model_name)]
    else:
        am = [LLMActionManager(mdp, 'player_0', layout_name, model_name),
              ScriptedActionManager(mdp, 'player_1', layout_name, scripted_collab)]

    #pdb.set_trace()
    if reflections != '':
        am[0].llm_agent.message.extend([
            {"role": "user", "content": reflections},
            {"role": "assistant", "content": "Thank you for this useful context!"}
        ])
        
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

    #pdb.set_trace()
    if reflections != '':
        reflexion_agent = LLMActionManager(mdp, 'player_0', layout_name, model_name, reflector=True)
        prev_game_results = 'My action history: ' + am[0].llm_agent.self._add_history()
        prev_game_results += '\nMy teammate\'s action history: ' + am[1].llm_agent.self._add_history()
        prev_game_results += '\nFinal score: ' + score

        new_reflection = 'Reflection on previous match: ' + reflexion_agent.get_reflection(prev_game_results)
        
    return score, new_reflection

#def _read_history(self):
#    # Check if action history exists in the log
#    if "action_history" in self.log_csv_dict:
#        # Retrieve the action history from the log
#        action_history = self.log_csv_dict["action_history"]
#        # Format it as a string, joining the list items
#        return f"Action history: {', '.join(action_history)}."
#    else:
#        return "No action history found."


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

    if reflexion == "true":
        reflections = 'Reflections on previously played games with this partner include:\n'
    else:
        reflections = ''
    
    for layout_name in LAYOUTS:
        scores = []
        gpt_3_costs = []
        gpt_4_costs = []
        for idx in range(NUM_TRIALS):
            score, prev_game_reflection = main(layout_name, model_name, scripted_collab, reflections)
            scores.append(score)
            reflections += f"Game {idx+1}: {prev_game_reflection}\n"

        with open(f'{layout_name}.txt', 'w') as f:
            f.write("MODEL: GPT4-turbo",)
            f.write(f"MEAN SCORE: {np.mean(scores)}\n")
            f.write(f"STD ERROR: {np.std(np.array(scores)) / np.sqrt(NUM_TRIALS)}\n")
            f.write(f"SAMPLE SCORES: {scores}\n")

        # test with cramped_room only for now, as it includes all possible actions for agents
        break
    
