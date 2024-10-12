import argparse
import numpy as np
import time
import os
import logging
from tqdm import tqdm

# Import necessary modules from the individual benchmark scripts
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from llm_coordination_agents.overcooked_action_manager import LLMActionManager
from overcooked_ai_py.mdp.actions import Action

from hanabi_learning_environment import pyhanabi
from llm_coordination_agents.hanabi_action_manager import run_game

from llm_coordination_agents.CollabEscapeMDP import Game
from llm_coordination_agents.collab_escape_agent import LLMAgent as CollabEscapeLLMAgent

from llm_coordination_agents.collab_capture_action_manager import Environment, Agent, Thief, GreedyAgent


# Setup logging configuration
def setup_logging(game_name):
    log_filename = f"{game_name}_benchmark.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )


# Function to save results to a file
def save_results(game_name, results):
    results_dir = 'benchmark_results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(results_dir, f"{game_name}_results_{timestamp}.txt"), 'w') as file:
        file.write(results)
    logging.info(f"Results saved to {game_name}_results_{timestamp}.txt")


# Overcooked Benchmark
def run_overcooked_benchmark(layout_name, model_name, num_trials):
    setup_logging("overcooked")
    logging.info(f"Starting Overcooked Benchmark on layout: {layout_name} with model: {model_name}")

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    scores = []
    
    for trial in range(num_trials):
        logging.info(f"Trial {trial+1}/{num_trials} for Overcooked Layout: {layout_name}")
        am = [
            LLMActionManager(mdp, 'player_0', layout_name, model_name),
            LLMActionManager(mdp, 'player_1', layout_name, model_name)
        ]
        state = mdp.get_standard_start_state()
        score = 0
        NUM_TICKS = 400
        
        for tick in tqdm(range(NUM_TICKS)):
            joint_action = [Action.STAY] * 2
            for i in range(2):
                action, _ = am[i].get_next_move(state, '')
                joint_action[i] = action
            
            state, sparse_reward, _ = mdp.get_state_transition(state, joint_action)
            score += sparse_reward
            
            if tick % 50 == 0:
                logging.info(f"Tick {tick}: Current Score: {score}")
        
        logging.info(f"Trial {trial+1} Score: {score}")
        scores.append(score)

    avg_score = np.mean(scores)
    std_error = np.std(scores) / np.sqrt(num_trials)
    results = f"Overcooked {layout_name} - Mean Score: {avg_score}, Std Error: {std_error}, Scores: {scores}"
    logging.info(results)
    save_results("overcooked", results)
    return scores


# Hanabi Benchmark
def run_hanabi_benchmark(model_name, seeds):
    setup_logging("hanabi")
    logging.info(f"Starting Hanabi Benchmark with model: {model_name}")

    scores = []
    assert pyhanabi.cdef_loaded() and pyhanabi.lib_loaded(), "PyHanabi failed to load"
    
    for seed in seeds:
        logging.info(f"Running Hanabi game with seed: {seed}")
        score = run_game({"players": 2, "random_start_player": False, "seed": seed}, model_name)
        scores.append(score)
        logging.info(f"Score for seed {seed}: {score}")

    avg_score = np.mean(scores)
    std_error = np.std(scores) / np.sqrt(len(seeds))
    results = f"Hanabi - Mean Score: {avg_score}, Std Error: {std_error}, Scores: {scores}"
    logging.info(results)
    save_results("hanabi", results)
    return scores


# Collab Escape Benchmark
def run_collab_escape_benchmark(model_name, num_trials):
    setup_logging("collab_escape")
    logging.info(f"Starting Collab Escape Benchmark with model: {model_name}")

    results = []
    turn_counts = []
    
    for trial in range(num_trials):
        logging.info(f"Trial {trial+1}/{num_trials} for Collab Escape")
        game = Game()
        outcome, turns = game.play(model_name)
        results.append(outcome)
        turn_counts.append(turns)
        logging.info(f"Trial {trial+1} Outcome: {outcome}, Turns: {turns}")
        
    wins = results.count('win')
    escape_rate = (wins / num_trials) * 100
    avg_turns = np.mean(turn_counts)
    std_error = np.std(turn_counts) / np.sqrt(num_trials)
    results_summary = f"Collab Escape - Escape Rate: {escape_rate}%, Mean Turns: {avg_turns}, Std Error: {std_error}, Turn Counts: {turn_counts}"
    logging.info(results_summary)
    save_results("collab_escape", results_summary)
    return results, turn_counts


# Collab Capture Benchmark
def run_collab_capture_benchmark(model_name, num_trials):
    setup_logging("collab_capture")
    logging.info(f"Starting Collab Capture Benchmark with model: {model_name}")

    turn_counts = []

    for trial in range(num_trials):
        logging.info(f"Trial {trial+1}/{num_trials} for Collab Capture")
        environment = Environment()
        alice = Agent(1, 0, "Alice", environment, model_name)
        bob = Agent(6, 1, "Bob", environment, model_name)
        thief = Thief(2, "Thief", environment)
        
        num_turns = 0
        MAX_TURNS = 30
        while num_turns < MAX_TURNS:
            state_for_llm = environment.get_state_for_llm(alice, bob, thief)
            alice_action = alice.llm_agent.get_next_move(state_for_llm)
            bob_action = bob.llm_agent.get_next_move(state_for_llm)
            
            if isinstance(alice_action, int):
                alice.plan_move(alice_action)
            elif alice_action.startswith("Press"):
                alice.plan_press_button()

            if isinstance(bob_action, int):
                bob.plan_move(bob_action)
            elif bob_action.startswith("Press"):
                bob.plan_press_button()

            thief.plan_move_away_from_agents(alice.next_room, bob.next_room)

            alice.execute_move()
            bob.execute_move()
            thief.execute_move()
            
            if num_turns % 10 == 0:
                logging.info(f"Turn {num_turns}: Alice at {alice.current_room}, Bob at {bob.current_room}, Thief at {thief.current_room}")
            
            num_turns += 1
            
            if (thief.current_room == alice.current_room) or (thief.current_room == bob.current_room):
                logging.info(f"Thief caught in {num_turns} turns.")
                break

        turn_counts.append(num_turns)
        logging.info(f"Trial {trial+1} completed in {num_turns} turns")
    
    avg_turns = np.mean(turn_counts)
    std_error = np.std(turn_counts) / np.sqrt(num_trials)
    results = f"Collab Capture - Mean Turns: {avg_turns}, Std Error: {std_error}, Turn Counts: {turn_counts}"
    logging.info(results)
    save_results("collab_capture", results)
    return turn_counts


# Main function to handle argument parsing and function calls
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run various game benchmarks with specific models")
    parser.add_argument('game', type=str, choices=['overcooked', 'hanabi', 'collab_escape', 'collab_capture'], help='The game to benchmark')
    parser.add_argument('model_name', type=str, help='The name of the model to benchmark')
    parser.add_argument('--layout_name', type=str, help='Layout name for Overcooked benchmark')
    parser.add_argument('--seeds', nargs='+', type=int, help='Seeds for Hanabi benchmark')
    parser.add_argument('--num_trials', type=int, default=3, help='Number of trials for benchmarking')

    args = parser.parse_args()

    if args.game == 'overcooked':
        if not args.layout_name:
            parser.error("--layout_name is required for the Overcooked game")
        run_overcooked_benchmark(args.layout_name, args.model_name, args.num_trials)
    elif args.game == 'hanabi':
        if not args.seeds:
            parser.error("--seeds are required for the Hanabi game")
        run_hanabi_benchmark(args.model_name, args.seeds)
    elif args.game == 'collab_escape':
        run_collab_escape_benchmark(args.model_name, args.num_trials)
    elif args.game == 'collab_capture':
        run_collab_capture_benchmark(args.model_name, args.num_trials)
    else:
        logging.error("Invalid or missing arguments for the specified game.")
