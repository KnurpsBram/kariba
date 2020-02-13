```python
from kariba_moismcts import Kariba, moismcts, Simulators

kariba = Kariba()
event = kariba.random_card_draw()
kariba.apply_event(event)
root_state = kariba
best_action = moismcts(root_state, n=100)
```

    100%|██████████| 100/100 [00:00<00:00, 174.21it/s]



```python
print(kariba)
print(event)
print(best_action)
```

    -------------------------
    turn: player0
    deck:
    [6 8 7 8 8 8 8 6]
    field:
    [0 0 0 0 0 0 0 0]
    hands:
    player0 [2 0 1 0 0 0 0 2]
    player1 [0 0 0 0 0 0 0 0]
    -------------------------
    
    {'kind': 'deck_draw', 'who': 'player0', 'cards': array([2, 0, 1, 0, 0, 0, 0, 2])}
    {'kind': 'action', 'who': 'player0', 'cards': array([0, 0, 0, 0, 0, 0, 0, 1])}



```python
# what happens inside the moismcts function
simulators = Simulators(root_state)

event = simulators.random_card_draw()
simulators.apply_event(event) 

print("Full information of the current game state:")
print(simulators.game)

print("Partial information available to player0:")
print(simulators.tree_dict["player0"].game)

print("Partial information available to player1:")
print(simulators.tree_dict["player1"].game)
```

    Full information of the current game state:
    -------------------------
    turn: player0
    deck:
    [6 8 7 8 8 8 8 6]
    field:
    [0 0 0 0 0 0 0 0]
    hands:
    player0 [2 0 1 0 0 0 0 2]
    player1 [0 0 0 0 0 0 0 0]
    -------------------------
    
    Partial information available to player0:
    -------------------------
    turn: player0
    deck:
    [6 8 7 8 8 8 8 6]
    field:
    [0 0 0 0 0 0 0 0]
    hands:
    player0 [2 0 1 0 0 0 0 2]
    player1 [0 0 0 0 0 0 0 0]
    -------------------------
    
    Partial information available to player1:
    -------------------------
    turn: player0
    deck:
    [6 8 7 8 8 8 8 6]
    field:
    [0 0 0 0 0 0 0 0]
    hands:
    player0 [2 0 1 0 0 0 0 2]
    player1 [0 0 0 0 0 0 0 0]
    -------------------------
    



```python
# run 6 simulations
for _ in range(6):
    while not simulators.game.is_final:
        simulators.apply_event(simulators.random_card_draw()) # give cards to the player whose turn it is, at the very first turn, this should not do anything
        simulators.apply_event(simulators.select_action()) # the player whose turn it is may select the action, apply the action to the game and update both the players' trees
        simulators.next_turn()
    winner = simulators.game.leading_player
    simulators.backpropagate(winner)
    simulators.reset_game()
```


```python
print("Tree of player0 after 6 simulations:")
print(simulators.tree_dict["player0"])

print("Tree of player1 after 6 simulations:")
print(simulators.tree_dict["player1"])
```

    Tree of player0 after 6 simulations:
    -------------------------
    self: player0
    n: 6
    jungle:
    [6 8 7 8 8 8 8 6]
    field:
    [0 0 0 0 0 0 0 0]
    hand:
    [2 0 1 0 0 0 0 2]
    -------------------------
        -------------------------
        self: player0
        n: 1
        w: 1
        jungle:
        [6 8 7 8 8 8 8 6]
        field:
        [1 0 0 0 0 0 0 0]
        hand:
        [1 0 1 0 0 0 0 2]
        -------------------------
            -------------------------
            self: player0
            n: 1
            w: 1
            jungle:
            [6 8 7 8 8 8 8 6]
            field:
            [1 0 0 0 0 0 0 0]
            hand:
            [1 0 1 0 0 0 0 2]
            -------------------------
            
            
        
        -------------------------
        self: player0
        n: 1
        w: 1
        jungle:
        [6 8 7 8 8 8 8 6]
        field:
        [0 0 0 0 0 0 0 2]
        hand:
        [2 0 1 0 0 0 0 0]
        -------------------------
        
        
        -------------------------
        self: player0
        n: 1
        w: 1
        jungle:
        [6 8 7 8 8 8 8 6]
        field:
        [0 0 1 0 0 0 0 0]
        hand:
        [2 0 0 0 0 0 0 2]
        -------------------------
        
        
        -------------------------
        self: player0
        n: 1
        w: 1
        jungle:
        [6 8 7 8 8 8 8 6]
        field:
        [0 0 0 0 0 0 0 1]
        hand:
        [2 0 1 0 0 0 0 1]
        -------------------------
        
        
        -------------------------
        self: player0
        n: 1
        w: 0
        jungle:
        [6 8 7 8 8 8 8 6]
        field:
        [2 0 0 0 0 0 0 0]
        hand:
        [0 0 1 0 0 0 0 2]
        -------------------------
        
        
    
    Tree of player1 after 6 simulations:
    -------------------------
    self: player1
    n: 6
    jungle:
    [8 8 8 8 8 8 8 8]
    field:
    [0 0 0 0 0 0 0 0]
    hand:
    [0 0 0 0 0 0 0 0]
    -------------------------
        -------------------------
        self: player1
        n: 1
        jungle:
        [7 8 8 8 8 8 8 8]
        field:
        [1 0 0 0 0 0 0 0]
        hand:
        [0 0 0 0 0 0 0 0]
        -------------------------
            -------------------------
            self: player1
            n: 1
            jungle:
            [7 8 8 8 8 8 8 8]
            field:
            [1 0 0 0 0 0 0 0]
            hand:
            [0 0 0 0 0 0 0 0]
            -------------------------
            
            
        
        -------------------------
        self: player1
        n: 1
        jungle:
        [8 8 8 8 8 8 8 6]
        field:
        [0 0 0 0 0 0 0 2]
        hand:
        [0 0 0 0 0 0 0 0]
        -------------------------
        
        
        -------------------------
        self: player1
        n: 1
        jungle:
        [8 8 7 8 8 8 8 8]
        field:
        [0 0 1 0 0 0 0 0]
        hand:
        [0 0 0 0 0 0 0 0]
        -------------------------
        
        
        -------------------------
        self: player1
        n: 1
        jungle:
        [8 8 8 8 8 8 8 7]
        field:
        [0 0 0 0 0 0 0 1]
        hand:
        [0 0 0 0 0 0 0 0]
        -------------------------
        
        
        -------------------------
        self: player1
        n: 1
        jungle:
        [6 8 8 8 8 8 8 8]
        field:
        [2 0 0 0 0 0 0 0]
        hand:
        [0 0 0 0 0 0 0 0]
        -------------------------
        
        
    



```python

```
