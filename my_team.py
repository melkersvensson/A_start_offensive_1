import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


def create_team(first_index, second_index, is_red,
                first='OffensiveAStarAgent', second='OffensiveAStarAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

class OffensiveAStarAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.powered_up_counter = 0
        self.deadEndCounter = 0
        self.deadEnds = set()

    
    def choose_action(self, gameState):
        """
        Determines the next action for the agent:
        - Handles powered-up behavior.
        - Targets capsules to enter powered-up mode.
        - Collects food and manages return-home logic.
        """
        if (not(self.deadEndCounter)):
            self.deadEnds = self.find_dead_ends(gameState)
            print(f"These are the dead end locations: {self.deadEnds}")
            self.deadEndCounter = 1
        # Check if powered-up mode is active
        if self.powered_up_counter > 0:
            print(f"Agent {self.index}: Powered-up mode active ({self.powered_up_counter} moves remaining).")
            food = self.get_food(gameState).as_list()
            my_food = self.divide_food(food, gameState)
            
            # If only one food remains, switch to defense or return-home mode
            if len(my_food) <= 3:
                print(f"Agent {self.index}: Only one food left. Returning home during powered-up mode.")
                if not self.on_own_side(gameState.get_agent_position(self.index), gameState):
                    return self.return_home(gameState)
                else:
                    return self.we_defend(gameState)
            
            # Continue powered-up behavior
            self.powered_up_counter -= 1
            return self.a_star_search(gameState, gameState.get_agent_position(self.index), my_food, False)

        # Check for capsules to trigger powered-up mode
        capsules = self.get_capsules(gameState)
        current_pos = gameState.get_agent_position(self.index)
        if current_pos in capsules:
            print(f"Agent {self.index}: Capsule eaten! Entering powered-up mode.")
            self.powered_up_counter = 40  # Activate powered-up mode
            return self.a_star_search(gameState, gameState.get_agent_position(self.index), my_food, False)

        # Normal offensive behavior
        food = self.get_food(gameState).as_list()
        my_food = self.divide_food(food, gameState)

        # If only one food remains, return home or switch to defense
        if len(my_food) <= 3:
            print(f"Agent {self.index}: Returning home.")
            if not self.on_own_side(gameState.get_agent_position(self.index), gameState):
                return self.return_home(gameState)
            else:
                return self.we_defend(gameState)

        # Otherwise, continue food collection using A* search
        return self.a_star_search(gameState, gameState.get_agent_position(self.index), my_food)


    def get_successor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor


    def a_star_search(self, gameState, start_pos, food, avoid_ghosts=True):
        frontier = util.PriorityQueue()
        frontier.push((start_pos, []), 0)  # (current position, path)
        explored = set()

        if (avoid_ghosts == False):
            self.powered_up_counter -= 1

        while not frontier.is_empty():
            current_pos, path = frontier.pop()

            # Mark state as explored
            if current_pos in explored:
                continue
            explored.add(current_pos)

            # Debug current state
            print(f"Exploring State: {current_pos}, Path So Far: {path}")

            # Goal check: Stop if the agent reaches food
            if current_pos in food:
                print(f"Goal Found! Reached {current_pos} with Path: {path}")
                return path[0] if path else Directions.STOP

            # Get legal actions and expand successors
            legal_actions = gameState.get_legal_actions(self.index)
            print(f"Legal Actions from {current_pos}: {legal_actions}")

            # Evaluate each action
            best_action = None
            best_heuristic = float('inf')
            for action in legal_actions:
                successor = self.get_successor(gameState, action)
                new_pos = successor.get_agent_position(self.index)

                # Avoid revisiting explored states
                if new_pos in explored:
                    continue

                h = self.heuristic(new_pos, food, gameState)

                print(f"Action: {action}, Successor State: {new_pos}, Heuristic: {h}")

                # Update best action and heuristic
                if h < best_heuristic:
                    best_heuristic = h
                    best_action = action

                # Add successor to the frontier
                g = len(path) + 1  # Path cost
                frontier.push((new_pos, path + [action]), g + h)

            # Return the best action at this state
            if best_action:
                print(f"Best Action from {current_pos}: {best_action} with Heuristic: {best_heuristic}")
                return best_action

        # If no valid path is found, return STOP
        print(f"Agent {self.index}: No valid path found.")
        return Directions.STOP


    def return_home(self, gameState):
        """
        Navigate back to the middle part of the first column of the home quadrant while avoiding ghosts.
        """
        current_pos = gameState.get_agent_position(self.index)
        home_target = self.get_home_target(gameState)  # Get validated home target

        # Initialize variables
        legal_actions = gameState.get_legal_actions(self.index)
        best_action = None
        best_heuristic = float('inf')

        # Debug current position and legal actions
        print(f"Agent {self.index}: Returning Home - Current Position: {current_pos}")
        print(f"Agent {self.index}: Legal Actions: {legal_actions}")

        # Iterate over all legal actions to find the best one
        for action in legal_actions:
            successor = self.get_successor(gameState, action)
            new_pos = successor.get_agent_position(self.index)

            # Compute heuristic
            h = self.return_home_heuristic(new_pos, home_target, gameState)
            print(f"Action: {action}, Successor State: {new_pos}, Heuristic: {h}")

            # Update best action based on heuristic
            if h < best_heuristic:
                best_heuristic = h
                best_action = action

        # Return the action with the lowest heuristic
        if best_action is not None:
            print(f"Agent {self.index}: Best Action: {best_action} with Heuristic: {best_heuristic}")
            return best_action
        else:
            print(f"Agent {self.index}: No valid action found, defaulting to STOP.")
            return Directions.STOP

    def we_defend(self, gameState):
        """
        Defensive behavior to move towards enemy Pacmans within the home quadrant.
        """
        legal_actions = gameState.get_legal_actions(self.index)

        # Track best action based on features
        best_action = None
        best_score = float('-inf')

        numberOfInvaiders = 0 
        for action in legal_actions:
            # Extract features for this action
            features = self.get_features(gameState, action)
            score = self.evaluate_defensive_features(features)
            
            if (features['num_invaders'] != 0):
                numberOfInvaiders = 1 
            # Update best action
            if score > best_score:
                best_score = score
                best_action = action

        # Default to STOP if no better action is found
        if (numberOfInvaiders == 1):
            return best_action if best_action else Directions.STOP
        else: 
            # go to center 
            return self.move_to_center(gameState)
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features
    
    def move_to_center(self, gameState):
        """
        Move towards the center of the home quadrant.
        """
        # Define the center position of the home quadrant
        center = self.get_home_target(gameState)
        current_pos = gameState.get_agent_position(self.index)

        # Use A* or another method to find the next action
        legal_actions = gameState.get_legal_actions(self.index)
        best_action = None
        best_dist = float('inf')

        for action in legal_actions:
            successor = self.get_successor(gameState, action)
            successor_pos = successor.get_agent_position(self.index)
            dist_to_center = self.get_maze_distance(successor_pos, center)

            if dist_to_center < best_dist:
                best_dist = dist_to_center
                best_action = action

        # Return the best action to move towards the center
        return best_action if best_action else Directions.STOP

    def evaluate_defensive_features(self, features):
        """
        Assign weights to features for evaluating the best defensive action.
        """
        weights = {
            'on_defense': 100,           # Encourage staying on defense
            'num_invaders': -1000,       # High priority for reducing invaders
            'invader_distance': -10,     # Prefer actions closer to invaders
            'stop': -500,                # Penalize stopping
            'reverse': -2                # Slightly penalize reversing direction
        }

        # Calculate weighted score
        score = sum(features[feature] * weight for feature, weight in weights.items())
        return score
    
    def defensive_heuristic(self, pos, invaders, gameState):
        """
        Simplified heuristic using features extracted for defensive behavior.
        """
        if invaders:
            distances = [self.get_maze_distance(pos, invader.get_position()) for invader in invaders]
            return min(distances)  # Heuristic: distance to the closest invader
        else:
            # Default heuristic to patrol the center of the home quadrant
            return self.get_maze_distance(pos, self.get_home_target(gameState))

    def get_home_target(self, gameState):
        """
        Calculate the central position on the border of the home side (your quadrant).
        The position should be the midpoint along the y-axis on your side where no wall exists.
        """
        walls = gameState.get_walls()
        width, height = gameState.data.layout.width, gameState.data.layout.height

        # Determine the border column on your side
        is_red = self.red
        border_x = width // 2 - 3 if is_red else width // 2 + 3  # Left border for red, right for blue

        # Define the y-midpoint based on the agent's role (full, top, or bottom half)
        if self.food_left_for_teammate(gameState):
            mid_y = height // 2
        elif self.index == self.get_team(gameState)[0]:  # Bottom-half teammate
            mid_y = height // 4
        else:  # Top-half teammate
            mid_y = 3 * height // 4

        # Check for non-wall positions progressively farther from the midpoint
        for offset in range(height):
            # Check upwards and downwards
            up_y = mid_y + offset
            down_y = mid_y - offset
            if (self.index == self.get_team(gameState)[0]):
                if up_y < height and not walls[border_x][up_y]:
                    return (border_x, up_y)
                if down_y >= 0 and not walls[border_x][down_y]:
                    return (border_x, down_y)
            else: 
                if down_y >= 0 and not walls[border_x][down_y]:
                    return (border_x, down_y)
                if up_y < height and not walls[border_x][up_y]:
                    return (border_x, up_y)
        # If no non-wall positions are found (very unlikely), default to the midpoint
        return (border_x, mid_y)


# strumpor, kalsonger, jeans (inte fina), handukar, etc...

    def return_home_heuristic(self, pos, home_target, gameState):
        """
        Heuristic for returning home:
        - Distance to the home target.
        - High penalty for being near ghosts.
        """
        try:
            # Base heuristic: Maze distance to the home target
            distance = self.get_maze_distance(pos, home_target)
        except Exception as e:
            # If positions are not valid, return a high heuristic
            print(f"Agent {self.index}: Invalid position in grid - {e}")
            return float('inf')

        # Ghost avoidance
        ghost_penalty = 0
        ghost_positions = [gameState.get_agent_position(opp) for opp in self.get_opponents(gameState)]
        for ghost in ghost_positions:
            if ghost:
                ghost_dist = self.get_maze_distance(pos, ghost)
                if (ghost_dist <= 4 and ghost_dist != 0):  # Avoid ghosts within 4 squares
                    ghost_penalty += 1000 / max(ghost_dist,1) # High penalty for close ghosts

        return distance + ghost_penalty

    def heuristic(self, pos, food, gameState):
        """
        Updated heuristic:
        - Prioritizes food collection and capsule targeting.
        - Avoids active ghosts, factoring in scared timers.
        - Penalizes dead ends when active ghosts are near.
        - Encourages chasing Pacman ghosts on our side.
        """
        # Closest food distance
        food_dist = min([self.get_maze_distance(pos, pellet) for pellet in food]) if food else float('inf')

        # Ghost avoidance and handling
        ghost_penalty = 0
        chase_bonus = 0  # Bonus for chasing Pacman ghosts
        ghosts = [gameState.get_agent_state(opp) for opp in self.get_opponents(gameState)]

        # Categorize ghosts
        active_ghosts = []
        scared_ghosts = []
        pacman_ghosts = []  # Opponents that are Pacman on our side

        for ghost in ghosts:
            if ghost.get_position() is not None:
                if ghost.scared_timer > 0:  # Ghost is scared
                    scared_ghosts.append(ghost.get_position())
                elif ghost.is_pacman:  # Ghost is Pacman
                    pacman_ghosts.append(ghost.get_position())
                else:  # Active, dangerous ghost
                    active_ghosts.append(ghost.get_position())

        # Ghost penalty and chase bonus
        if self.on_own_side(pos, gameState):
            # Small bonus for chasing Pacman ghosts
            chase_bonus = 0.2 * sum(1 / (1 + self.get_maze_distance(pos, ghost)) for ghost in pacman_ghosts if ghost)
        else:
            # Penalize proximity to active (dangerous) ghosts
            ghost_penalty = 2*sum(1 / (1 + self.get_maze_distance(pos, ghost)) for ghost in active_ghosts if ghost)
            - sum(1 / (1 + self.get_maze_distance(pos, ghost)) for ghost in scared_ghosts if ghost)
        # Capsule prioritization
        capsules = self.get_capsules(gameState)
        capsule_bonus = 0
        if capsules:
            closest_capsule_dist = min(self.get_maze_distance(pos, capsule) for capsule in capsules)
            capsule_bonus = 200 / (1 + closest_capsule_dist)  # Reward proximity to capsules

        # Dead-end penalty (only if active ghosts are present)
        deadEndPenalty = 0
        if pos in self.deadEnds and len(active_ghosts) > 0:  # Penalize only with active ghosts around
            min_ghost_dist = min(self.get_maze_distance(pos, ghost) for ghost in active_ghosts if ghost)
            if min_ghost_dist < 5:  # Adjust threshold as needed
                deadEndPenalty = 10 / (1 + min_ghost_dist)  # Scales with proximity

        # Combine heuristic components
        return (
            1 * food_dist
            - 3 * capsule_bonus
            + 15 * ghost_penalty
            - chase_bonus  # Encourage chasing Pacman ghosts
            + 5 * deadEndPenalty
        )


    def divide_food(self, food, gameState):
        # Divide food into two hemispheres based on board dimensions
        width, height = gameState.data.layout.width, gameState.data.layout.height
        mid = height // 2  # Divide by height for top/bottom division

        indeces = self.get_team(gameState)
        check = True
        if(self.index == indeces[0]):
            check = False

        if check == True:  # Agent 0, 2, etc.
            my_food = [f for f in food if f[1] < mid]  # Bottom half first index
        else:  # Agent 1, 3, etc.
            my_food = [f for f in food if f[1] >= mid]  # Top half

        # Debugging
        print(f"Agent {self.index}: Assigned food -> {my_food}")
        return my_food
    
    def food_left_for_teammate(self, gameState):
        food_left_to_eat = len(self.get_food(gameState).as_list())
        if (food_left_to_eat > 6):
            return True
        else:
            return False 

    def find_dead_ends(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        dead_ends = set()  # Set to store coordinates of dead ends
        
        # Iterate over the grid
        for x in range(width):
            for y in range(height):
                if not gameState.has_wall(x, y):  # Check only passable tiles
                    if self.count_accessible_neighbors(x, y, width, height, gameState) == 1:  # Only one way out
                        dead_ends.add((x, y))
        
        return dead_ends

    # Helper function to check neighbors
    def count_accessible_neighbors(self, x, y, width, height, gameState):
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return sum(
            1
            for nx, ny in neighbors
            if 0 <= nx < width and 0 <= ny < height and not gameState.has_wall(nx, ny)
        )
    
    def on_own_side(self, pos, gameState):
        mid = gameState.data.layout.width // 2
        return pos[0] < mid if self.red else pos[0] >= mid
