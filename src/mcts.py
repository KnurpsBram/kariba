import sys
import copy
import numpy as np
import random
import itertools
import tqdm

import util
#
# from src.kariba import Kariba
# from src.util import NodenameGenerator
#
# nodename_generator = NodenameGenerator()

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

class Kariba():
    def __init__(self, deck=None, field=None, hands=None, whose_turn_=0, player_names=["player0","player1"], n_species=8, max_n_hand=5):
        self.n_species = n_species
        self.max_n_hand = max_n_hand

        self.whose_turn_  = whose_turn_
        self.player_names = player_names
        self.n_players    = len(player_names)

        self.deck       = np.ones(self.n_species, dtype=int) * max(3, self.n_species) if deck is None else deck
        self.field      = np.zeros(self.n_species, dtype=int)
        self.hands      = {player : np.zeros(self.n_species, dtype=int) for player in self.player_names}
        self.scoreboard = {player : 0 for player in self.player_names}

    @property
    def whose_turn(self):
        return self.player_names[self.whose_turn_]

    @property
    def who_next_turn_(self):
        return (self.whose_turn_ + 1) % self.n_players

    @property
    def is_final(self):
        return np.sum(self.deck) == 0 and (any([np.sum(self.hands[player])==0 for player in self.player_names]))

    @property
    def leading_player(self):
        return util.keywithmaxval(self.scoreboard)

    def next_turn(self):
        self.whose_turn_ = self.who_next_turn_

    def jungle(self, player):
        return sum([self.deck] + [hand for player_, hand in self.hands.items() if player_ != player])

    def apply_event(self, event):
        if event["kind"] == "deck_draw":
            self.deck                -= event["cards"]
            self.hands[event["who"]] += event["cards"]

        if event["kind"] == "action":
            self.hands[event["who"]] -= event["cards"]
            self.field               += event["cards"]

            action_animal = np.nonzero(event["cards"])[0][0]
            if self.field[action_animal] >= 3:
                # we use fear_animal=-1 to denote a situation where there's no animals to be chased away
                fear_animal = self.n_species - 1 if action_animal == 0 else max([idx for idx in self.field.nonzero()[0] if idx < action_animal] + [-1])
                if fear_animal >= 0:
                    self.scoreboard[self.whose_turn] += self.field[fear_animal]
                    self.field[fear_animal] = 0

    def allowed_actions(self, player):
        hand = self.hands[player]
        cards_list = [n*util.one_hot(idx, n_dim=self.n_species) for idx in range(self.n_species) for n in range(1, int(self.hands[player][idx])+1)]
        event_list = [{
            "kind"  : "action",
            "who"   : player,
            "cards" : cards
        } for cards in cards_list]
        return event_list

    def random_card_draw(self):
        n_hand = np.sum(self.hands[self.whose_turn])
        n_to_draw = min(self.max_n_hand - n_hand, np.sum(self.deck))

        cards = np.zeros(self.n_species, dtype=int)
        for _ in range(n_to_draw):

            cards += util.one_hot(np.random.choice(range(self.n_species), p=(self.deck-cards) / np.sum(self.deck-cards)), n_dim=self.n_species)

        event = {
            "kind"  : "deck_draw",
            "who"   : self.whose_turn,
            "cards" : cards
        }

        return event

    def __repr__(self):
        s = \
        "-------------------------\n" + \
        "turn: " + self.whose_turn + "\n" + \
        "deck:\n" + str(self.deck) + "\n"+ \
        "field:\n" + str(self.field) + "\n" + \
        "hands:\n"+"\n".join([name+" "+str(hand) for name, hand in self.hands.items()]) + "\n" + \
        "-------------------------\n"
        return s

def is_equivalent_node(node_a, node_b):
    return all([                                     \
        node_a.player == node_b.player,              \
        np.array_equal(node_a.hand,   node_b.hand),  \
        np.array_equal(node_a.field,  node_b.field), \
        np.array_equal(node_a.jungle, node_b.jungle) \
    ])

class Node():
    def __init__(self, game, player, event=None, parent=None, c=np.sqrt(2)):
        self.player   = player # an agent
        self.parent   = parent # a node
        self.children = []

        self.hand     = game.hands[self.player]
        self.field    = game.field
        self.jungle   = game.jungle(self.player)

        self.n = 0 # number of simulations run from this node

        # event is the event that transitioned the parent node to the current node
        self.is_root_node        = parent is None
        self.is_pre_action_node  = (event["kind"] == "deck_draw" and event["who"] == self.player) if event is not None else False or (self.is_root_node and game.whose_turn == self.player) # we model the game as if drawing cards happens at the start of the turn
        self.is_post_action_node = (event["kind"] == "action"    and event["who"] == self.player) if event is not None else False

        if self.is_pre_action_node:
            self.untried_actions = game.allowed_actions(self.player)
            random.shuffle(self.untried_actions)

        if self.is_post_action_node:
            self.action = event
            self.w = 0 # number of simulations won
            self.c = c # hyperparameter that determines the tradeoff between exploration and exploitation

    @property
    def ucb(self):
        if self.is_post_action_node:
            return (self.w / self.n) + self.c * np.sqrt(2*np.log(self.parent.n)/self.n) # what if n==0?

    def backpropagate(self, winner):
        self.n += 1
        if self.is_post_action_node:
            self.w += winner == self.player

        if self.parent is not None: # backpropagate all the way up to the root node. The root node doesn't have a parent
            self.parent.backpropagate(winner)

    def __repr__(self):
        s = \
        "-------------------------\n" + \
        "self: " + self.player + "\n" + \
        "n: " + str(self.n) + "\n" + \
        ("w: " + str(self.w) + "\n" if self.is_post_action_node else "") + \
        "jungle:\n" + str(self.jungle) + "\n"+ \
        "field:\n" + str(self.field) + "\n" + \
        "hand:\n" + str(self.hand) + "\n" + \
        "-------------------------\n"
        return s

class Tree():
    def __init__(self, game, player):
        self.game   = game # assign by reference. If the game changes outside, it changes inside as well
        self.player = player

        self.root_node    = Node(copy.deepcopy(game), player) # use deepcopy to ensure that if the game changes outside the node, the attributes of the node don't change
        self.current_node = self.root_node

        # during selection (self.is_on_rollout_policy=False), we select actions based on UCB and keep track of new nodes.
        # during rollout (self.is_on_rollout_policy=True), we select actions randomly and do NOT keep track of new nodes
        self.is_on_rollout_policy = False

    def select_action(self, return_best_action=False):
        if return_best_action: # the child with the highest number of visits
            assert self.current_node.is_pre_action_node, "the node that is supposed to dictate an action is not a pre-action-node"
            return self.current_node.children[np.argmax([child.n for child in self.current_node.children])].action
        else:
            if self.is_on_rollout_policy: # a random action
                return np.random.choice(self.game.allowed_actions(self.player))
            else: # try each action at least once, then select action with highest UCB
                assert self.current_node.is_pre_action_node, "the node that is supposed to dictate an action is not a pre-action-node"
                if len(self.current_node.untried_actions) > 0:
                    return self.current_node.untried_actions.pop()
                return self.current_node.children[np.argmax([child.ucb for child in self.current_node.children])].action

    def apply_event(self, event):
        if not self.is_on_rollout_policy:
            new_node = Node(copy.deepcopy(self.game), event=event, player=self.player, parent=self.current_node)
            if not is_equivalent_node(self.current_node, new_node):
                for child in self.current_node.children:
                    if is_equivalent_node(child, new_node):
                        self.current_node = child
                        pass
                self.current_node.children.append(new_node)
                self.current_node = new_node
                if self.current_node.is_post_action_node:
                    self.is_on_rollout_policy = True

    def backpropagate(self, winner):
        self.current_node.backpropagate(winner)

    def __repr__(self):
        def print_children(node): # recursion!
            return "\n".join([util.indent_string(child.__repr__()+print_children(child), indent_spaces=4) for child in node.children])+"\n"
        return self.root_node.__repr__() + print_children(self.root_node)

class Simulators():
    '''
    A class that keeps track of how the game proceeds as viewed by all
    entities that can have an influence on the game. These entities are player0, player1 and the game itself.

    Player0 can view the cards in its own hand, but not the hand of the opponent
    Player0 can perform an action and put cards from its hand to the field
    Player0 decides what actions to play based on UCB (random for rollout policy)
    Player0 can't control what cards to draw from the deck

    Player1 likewise

    The last entity is 'the game itself', it decides what cards to deal to the players
    '''
    def __init__(self, game):
        self.reset_state = copy.deepcopy(game)
        self.game      = game
        self.tree_dict = {player : Tree(self.game, player) for player in self.game.player_names}
        self.trees     = self.tree_dict.values()

    @property
    def whose_turn(self):
        return self.game.whose_turn

    def random_card_draw(self):
        return self.game.random_card_draw()

    def select_action(self, return_best_action=False):
        return self.tree_dict[self.whose_turn].select_action(return_best_action=return_best_action)

    def reset_game(self):
        self.game = copy.deepcopy(self.reset_state)
        for tree in self.trees:
            tree.game = self.game
            tree.current_node = tree.root_node
            tree.is_on_rollout_policy = False # switch to UCB-policy rather than rollout policy

    def apply_event(self, event):
        for simulator in (self.game, *self.trees):
            simulator.apply_event(event)

    def next_turn(self):
        self.game.next_turn()

    def backpropagate(self, winner):
        for tree in self.trees:
            tree.backpropagate(winner)

def moismcts(root_state, n=100):
    '''
    Multiple Observer Information Set Monte Carlo Tree Search (MOISMCTS)
    keeps a separate tree for each player in which the state is encoded according to what the player can observe
    '''

    simulators = Simulators(copy.deepcopy(root_state))

    for i in tqdm.tqdm(range(n)):

        while not simulators.game.is_final:
            simulators.apply_event(simulators.random_card_draw()) # give cards to the player whose turn it is, at the very first turn, this should not do anything
            simulators.apply_event(simulators.select_action()) # the player whose turn it is may select the action, apply the action to the game and update both the players' trees
            simulators.next_turn()

        winner = simulators.game.leading_player
        simulators.backpropagate(winner)
        simulators.reset_game()

    return simulators.select_action(return_best_action=True)

# kariba = Kariba()
# event = kariba.random_card_draw()
# kariba.apply_event(event)
# root_state = kariba
# best_action = moismcts(root_state, n=1000)
# print(root_state)
# print(best_action)
