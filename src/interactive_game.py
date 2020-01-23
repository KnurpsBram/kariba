import copy
import numpy as np
import IPython.display as ipd

from src import util
from src import kariba
from src import mcts
from src.kariba import Kariba

def interactive_game(n=100):



    human_name         = input("Okay Human! what is your name? ")
    show_opponent_hand = util.str_to_bool(input("Do you want the AI's cards to be visible to you? (y/n)"))
    show_deck          = util.str_to_bool(input("Do you want the contents of the deck to be visible to you? (y/n)"))

    ai_name      = "Monty Carlos"
    player_names = [human_name, ai_name]

    print("Okay, let's flip a coin to see who may begin the game")
    whose_turn = np.random.randint(2)
    print("Very well! ", player_names[whose_turn], " may begin! \n")

    kariba = Kariba(
        player_names = player_names,
        whose_turn   = whose_turn,
        show_hands   = [True, show_opponent_hand],
        show_deck    = show_deck,
        n_species    = 5
    )

    def get_action():
        action = kariba.action_str_to_arr(input("What's your move?"))

        

        if any([np.array_equal(action, allowed_action) for allowed_action in kariba.allowed_actions]):
            return action
        else:
            print("Erm. That's not a valid move")
            return get_action()

    # because this is a game for humans, deal cards before game
    for _ in range(len(player_names)):
        kariba.apply_deck_draw(kariba.random_deck_draw())
        kariba.next_turn()

    while not kariba.is_final:
        if kariba.whose_turn == 0:
            print("The Game State is Now: \n", kariba)

            action = get_action()

            ipd.clear_output()

        elif kariba.whose_turn == 1:
            print("The Game State is Now: \n", kariba)
            print(ai_name, "is planning its next move...")

            # make a copy of the kariba gamestate to one where the kariba cannot observe the human's hand
            kariba_copy = copy.deepcopy(kariba)
            hand_mask   = np.zeros_like(kariba_copy.hands)
            hand_mask[kariba_copy.whose_turn, :] = 1
            kariba_copy.deck  = kariba_copy.deck + np.sum(kariba_copy.hands * (1-hand_mask), axis=0)
            kariba_copy.hands = kariba_copy.hands * hand_mask

            root_node = mcts.FullHandNode(kariba_copy, nodename="ROOT")
            action = mcts.mcts(root_node, n=n)

            ipd.clear_output()

        kariba.apply_action(action, verbose=True)
        kariba.apply_deck_draw(kariba.random_deck_draw())
        kariba.next_turn()
        print("")

    print(player_names[kariba.leading_player], " won!")
    # TODO: print won/lost
