import copy
import numpy as np
import random
import itertools
import tqdm

from src.kariba import Kariba
from src.util import NodenameGenerator

nodename_generator = NodenameGenerator()

# class Node():
#     '''
#     A General Node Class
#
#     There can be two types of nodes: FullHandNodes and IncompleteHandNodes.
#     This class describes properties both nodes have, such as a name, parent and children
#     '''
#     def __init__(self, kariba, parent=None, children=[], nodename=None):
#         self.kariba   = kariba # an object containing the game state and functions that could be applied to it
#         self.parent   = parent
#         self.children = children
#         self.nodename = next(nodename_generator) if nodename is None else nodename
#
#     @property
#     def is_leaf(self):
#         return len(self.children) == 0
#
#     @property
#     def depth(self):
#         '''
#         returns the depth of the tree below this node
#         '''
#         return 1 + max([child.depth for child in self.children]) if not self.is_leaf else 1
#
#     @property
#     def node_repstr(self):
#         '''
#         A repstr (representation string) helps to construct what is printed when we call print on an object of this class
#         '''
#         parentname = str(self.parent.nodename) if self.parent is not None else "This node has no parent"
#         childnames = str([node.nodename for node in self.children])
#
#         s = \
#         "#################################\n"+\
#         "Nodename : "+self.nodename+"\n"+\
#         "Parent   : "+parentname+"\n"+\
#         "Children : "+childnames+"\n"
#         return s
#
#     def __repr__(self):
#         return self.node_repstr + self.kariba.kariba_repstr
#
# class FullHandNode(Node):
#     def __init__(self, kariba, deck_draw=None, **kwargs):
#         super().__init__(kariba, **kwargs)
#
#         self.deck_draw = deck_draw if deck_draw is not None else np.random.choice(self.kariba.possible_deck_draws)
#         self.kariba.apply_deck_draw(self.deck_draw)
#
#         self.likelihood = self.deck_draw["likelihood"]
#
#         self.n = 0 # amount of simulations from this node
#
#     def sprout_child(self, action):
#         child = IncompleteHandNode(
#             kariba   = copy.deepcopy(self.kariba),
#             action   = action,
#             parent   = self
#         )
#         return child
#
#     def expand(self):
#         self.children = [self.sprout_child(action) for action in self.kariba.allowed_actions]
#
#     def select_child(self):
#         return self.children[np.argmax([child.UCB for child in self.children])]
#
#     @property
#     def best_action(self):
#         '''
#         The best action is the action which we've simulated most
#         This is because we know most about this action,
#         besides, the reason that we've explored this action most is because the prospect of winning form this node must have been good
#         However, we don't want to use UCB for making a move in the game, because we want to focus on exploitation
#         '''
#         return self.children[np.argmax([child.n for child in self.children])].action
#
#     def rollout(self):
#         '''
#         Determine who is likely to win a game starting from the current state,
#         by simulating all players' actions by a rollout policy (typically fully random actions)
#         '''
#         simulated_game = copy.deepcopy(self.kariba)
#
#         while not simulated_game.is_final:
#             action = random.choice(simulated_game.allowed_actions) # TO DO: make 'rollout-policy' more formal to include a possible not-random one, but based on NNs
#             simulated_game.apply_action(action).next_turn()
#
#             possible_deck_draws = simulated_game.possible_deck_draws
#             if len(possible_deck_draws) > 0:
#                 random_deck_draw = random.choice(possible_deck_draws)
#                 simulated_game.apply_deck_draw(random_deck_draw)
#         return simulated_game.leading_player
#
#     def backpropagate(self, winner):
#         self.n += 1
#         if self.parent is not None:
#             self.parent.backpropagate(winner)
#
#     @property
#     def fullhandnode_repstr(self):
#         '''
#         A repstr (representation string) helps to construct what is printed when we call print on an object of this class
#         '''
#
#         s = \
#         "+++++++++++++++++++++++++++++++++\n"+\
#         "NodeType         : FullHandNode\n"+\
#         "Likelihood       : "+str(self.likelihood)+"\n"
#         return s
#
#     def __repr__(self):
#         return self.node_repstr + self.fullhandnode_repstr + self.kariba.kariba_repstr
#
# class IncompleteHandNode(Node):
#     def __init__(self, kariba, action, **kwargs):
#         super().__init__(kariba, **kwargs)
#
#         self.action = action
#
#         self.w = 0 # amount of simulations won from this node
#         self.n = 0 # amount of simulations from this node
#
#         # a hyperparameter controlling the confidence of the upper confidence bound
#         # higher values drive the search more toward exploration over exploitation
#         self.c = np.sqrt(10)
#
#     def sprout_child(self, kariba, deck_draw):
#         child = FullHandNode(
#             kariba     = copy.deepcopy(kariba),
#             deck_draw  = deck_draw,
#             parent     = self
#         )
#         return child
#
#     def expand(self):
#         '''
#
#         TODO: explain why you don't apply action to a StateAction node
#         and why that means you must do it here with all these deepcopy statements
#
#         '''
#         self_copy = copy.deepcopy(self)
#         self_copy.kariba.apply_action(self.action).next_turn()
#         self.children = [self.sprout_child(self_copy.kariba, deck_draw) for deck_draw in self_copy.kariba.possible_deck_draws]
#
#     def select_child(self):
#         likelihoods = np.array([child.likelihood for child in self.children])
#         likelihoods = likelihoods / sum(likelihoods)
#         return np.random.choice(self.children, p=likelihoods)
#
#     @property
#     def UCB(self): # Upper Confidence Bound
#         # if the node is untried, set the UCB to an absurdly high value to ensure that tree search will have to try a new node at least once
#         return (self.w / self.n) + self.c * np.sqrt(2*np.log(self.parent.n)/self.n) if self.n > 0 else 100
#
#     def backpropagate(self, winner):
#         self.n += 1
#         self.w += self.kariba.whose_turn == winner
#         self.parent.backpropagate(winner)
#
#     @property
#     def incompletehandnode_repstr(self):
#         s = \
#         "+++++++++++++++++++++++++++++++++\n"+\
#         "NodeType         : IncompleteHandNode\n"+\
#         "Action           : "+str(self.action)+"\n"+\
#         "Simulations from this action onward : "+str(self.n)+"\n"+\
#         "Simulations won from this action onward : "+str(self.w)+"\n"
#         return s
#
#     def __repr__(self):
#         return self.node_repstr + self.incompletehandnode_repstr + self.kariba.kariba_repstr
#
# def mcts(root_node, n=100):
#     for _ in tqdm.tqdm(range(n)):
#         node = root_node
#         depth = 0
#         while not node.kariba.is_final: # simulate games untill they end or untill a fixed depth is reached
#             depth += 1
#             while not node.is_leaf:
#                 node = node.select_child()
#             node.expand()
#             if isinstance(node, FullHandNode):
#                 break
#         winner = node.rollout()
#         node.backpropagate(winner)
#
#     best_action = root_node.best_action
#     return best_action

# # MOISMCTS Multiple-Observer Information Set Monte Carlo Tree Search
# def moismcts(root_node, n=100):
#     ## TEMPORARY PSUEDOCODE
#     trees = [Tree(player1), Tree(Player2)]
#     for _ in tqdm.tqdm(range(n)):
#         kariba = root_node.kariba
#         while not kariba.is_final:
#             deck_draw = kariba.draw_random_cards() # draw card for 1 player. only first time might do nothing
#             kariba.apply_deck_draw(deck_draw)
#             action = trees[kariba.whose_turn].select_action() # select best-UCB action, unless this is a node we know nothing of, then just be random
#             kariba.apply_action(action)
#             for tree in trees:
#                 # selecting a node means to either create a new node if it doesn't yet exist for the current state, (considering observability)
#                 # or selecting the one that does exists such that you can select one of its children based on UCB.
#                 # only create new nodes if the the parent node has made more simulations,
#                 # otherwise you'll create a really deep branch upon your very first simulations
#                 tree.next_node(kariba)
#             kariba.next_turn()
#         winner = kariba.leading_player
#         for tree in trees:
#             tree.backpropagate(winner)

# def moismcts(root_node, n=100):
#     trees = [Tree(player1), Tree(player2)]
#
#     for _ in tqdm.tqdm(range(n)):
#         while not kariba.is_final:
#
#             deck_draw = kariba.draw_random_cards()
#             kariba.apply_deck_draw(deck_draw)
#             for tree in trees:
#                 tree.apply_deck_draw(deck_draw)
#
#             action = trees[kariba.whose_turn].select_action()
#             kariba.apply_action(action)
#             for tree in trees:
#                 tree.apply_action(action)
#
#             kariba.next_turn()
#             for tree in trees:
#                 tree.next_turn()
#
#         winner = kariba.leading_player
#         for tree in trees:
#             tree.backpropagate(winner)

simulators = [game, Tree(player1), Tree(player2)]


class Simulators():
    '''
    A class that keeps track of how the game proceeds as viewed by all
    entities that have an influence. These entities are player0, player1 and the game itself.

    Player0 can view the cards in its own hand, but not the hand of the opponent
    Player0 can perform an action and put cards from its hand to the field
    Player0 decides what actions to play based on UCB (random for rollout policy)
    Player0 can't control what cards to draw from the deck

    Player1 likewise

    The last entity is 'the game itself', it decides what cards to deal to the players
    '''
    def __init__(self, game):
        self.reset_state = copy.deepcopy(game)
        self.game = game
        self.trees = [Tree(player) for player in self.game.player_names]

    @property
    def whose_turn(self):
        return self.game.whose_turn

    def select_action(self):
        return self.trees[self.whose_turn].select_action()

    def reset_game(self):
        self.game = copy.deepcopy(self.reset_state)

    def apply_event(self, event):
        for simulator in (self.game, *self.trees):
            simulator.apply_event(event)

    def next_turn(self, event):
        for simulator in (self.game, *self.trees):
            simulator.next_turn()

    def backpropagate(self, winner):
        for tree in self.trees:
            tree.backpropagate(winner)

def moismcts(root_state, n=100):
    '''
    Multiple Observer Information Set Monte Carlo Tree Search (MOISMCTS)
    keeps a separate tree for each player
    '''

    simulators = Simulators(root_state)

    for _ in tqdm.tqdm(range(n)):
        simulators.reset_game()

        while not simulators.game.is_final:
            simulators.apply_event(simulators.game.draw_random_cards())
            simulators.apply_event(simulators.select_action())
            simulators.next_turn()
        winner = simulators.game.leading_player
        simulators.backpropagate(winner)

event = {
    "what"  : "deck draw",
    "who"   : player0,
    "cards" : [0, 2, 0, 1, ....]
}

event = {
    "what"  : "action",
    "who"   : player1,
    "cards" : [0, 1, 0, 0, ...]
}
