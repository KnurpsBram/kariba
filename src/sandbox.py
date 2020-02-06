import copy
import numpy as np

class Game():
    def __init__(self):
        self.field = np.array([2, 2, 2])
        self.hand  = np.array([1, 1, 1])

    def apply_action(self, action):
        self.field += action
        self.hand  -= action

class Node():
    def __init__(self, game):
        # self.field = game.field
        # self.hand  = game.hand
        self.field = copy.deepcopy(game.field)
        self.hand  = copy.deepcopy(game.hand)

class Tree():
    def __init__(self, game):
        self.game = game
        self.root_node = Node(game)

    def apply_action(self, action):
        self.game.apply_action(action)

game = Game()
tree = Tree(game)

print("-----tree.root_node------")
print(tree.root_node.field)
print(tree.root_node.hand)
print("-----tree.game-----------")
print(tree.game.field)
print(tree.game.hand)

print("**********")
print("apply action")
print("**********")
action = np.array([0, 1, 0])
game.apply_action(action)

print("-----tree.root_node------")
print(tree.root_node.field)
print(tree.root_node.hand)
print("-----tree.game-----------")
print(tree.game.field)
print(tree.game.hand)
