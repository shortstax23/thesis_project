import pyspiel 
import numpy as np
from open_spiel.python.algorithms import cfr 
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools 
from open_spiel.python.algorithms import random_agent
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability

game = pyspiel.load_game("leduc_poker(players=2)")
state = game.new_initial_state()
solver = cfr.CFRSolver(game)

NUM_ITERATIONS = 1000
LOG_EVERY = 100

for i in range(1, NUM_ITERATIONS+1):
    solver.evaluate_and_update_policy()

    if i % LOG_EVERY == 0:
        avg_policy = solver.average_policy()
        expl = exploitability.exploitability(game, avg_policy)
        print(f"{i:>10}  {expl:>16.6f}")

average_policy = solver.average_policy()
final_expl = exploitability.exploitability(game, average_policy)
print(f"\nFinal exploitability after {NUM_ITERATIONS} iterations: {final_expl:.6f}")

def show_policy_for_state(state, policy, max_depth=3, depth=0):
    """
    Print the strategy at each decision point, up to max_depth levels.
    Avoids the exponential blowup of printing the full tree.
    """
    if depth > max_depth or state.is_terminal():
        return
 
    if state.is_chance_node():
        # Just follow the first chance outcome so we reach a real decision
        action, _ = state.chance_outcomes()[0]
        show_policy_for_state(state.child(action), policy, max_depth, depth + 1)
        return
 
    info_state = state.information_state_string()
    probs      = policy.action_probabilities(state)
    actions    = state.legal_actions()
    action_names = [state.action_to_string(state.current_player(), a) for a in actions]
 
    indent = "  " * depth
    print(f"{indent}Player {state.current_player()} | InfoState: {info_state}")
    for a, name, p in zip(actions, action_names, [probs[a] for a in actions]):
        print(f"{indent}  {name:10s}  p = {p:.4f}")
 
    # Recurse only along the highest-probability action to stay readable
    best_action = max(actions, key=lambda a: probs.get(a, 0))
    show_policy_for_state(state.child(best_action), policy, max_depth, depth + 1)
 
 
print("=" * 55)
print("Sample policy trace (following most-likely actions):")
print("=" * 55)
show_policy_for_state(game.new_initial_state(), average_policy, max_depth=5)
 

from open_spiel.python.algorithms import evaluate_bots
import random
 
class PolicyBot:
    """Minimal bot wrapper that samples from a CFR average policy."""
    def __init__(self, player_id, policy):
        self.player_id = player_id
        self._policy   = policy
 
    def step(self, state):
        probs   = self._policy.action_probabilities(state)
        actions = list(probs.keys())
        weights = [probs[a] for a in actions]
        return random.choices(actions, weights=weights)[0]
 
 
def simulate_games(n=500):
    bot0 = PolicyBot(0, average_policy)
    bot1 = PolicyBot(1, average_policy)
    returns = [0.0, 0.0]
 
    for _ in range(n):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                action = random.choices(outcomes, weights=probs)[0]
            else:
                bot = bot0 if state.current_player() == 0 else bot1
                action = bot.step(state)
            state.apply_action(action)
        r = state.returns()
        returns[0] += r[0]
        returns[1] += r[1]
 
    print(f"\nSimulated {n} hands (self-play with average policy):")
    print(f"  Avg return player 0: {returns[0]/n:+.4f}")
    print(f"  Avg return player 1: {returns[1]/n:+.4f}")
    print("  (Should be near 0 for both — zero-sum Nash play)\n")
 
simulate_games(500)
print("Done.")
 

