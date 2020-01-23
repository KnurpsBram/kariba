import copy
import numpy as np
import random
import itertools

from src import util

class Kariba:
    def __init__(self, deck=None, field=None, hands=None, whose_turn=0, player_names=["player1", "player2"], show_hands=None, show_deck=True, n_species=8, max_n_hand=5):
        '''
        We represent the game with 8-dimensional arrays
        Each index belongs to a certain animal
        0=mouse, 1=meerkat, 2=zebra, 3=giraffe, 4=ostrich, 5=leopard, 6=rhino, 7=elephant
        The array specifies the amount of each animal

        So if there are 2 zebras and 1 leopard on the field, we'd have
            self.field = [0 0 2 0 0 1 0 0]
        '''
        self.n_species = n_species
        self.max_n_hand = max_n_hand
        self.deck   =  np.ones(self.n_species, dtype=int) * max(3, self.n_species) if deck  is None else deck
        self.field  = np.zeros(self.n_species, dtype=int)     if field is None else field

        self.whose_turn   = whose_turn
        self.player_names = player_names
        self.n_players    = len(self.player_names)

        self.hands      = np.zeros((self.n_players, self.n_species), dtype=int) if hands is None else hands
        self.scoreboard = np.zeros( self.n_players    , dtype=int)

        self.show_hands = [True]*self.n_players if show_hands is None else show_hands # whether to show the hand of a specific player when print() is called on an object of this class
        assert len(self.show_hands) == self.n_players, "kariba.show_hands doesn't have the same amount of booleans as players"
        self.show_deck  = show_deck

        self.animal_names = [
            ("mouse", "mice"),
            ("meerkat", "meerkats"),
            ("zebra", "zebras"),
            ("giraffe", "giraffes"),
            ("ostrich", "ostriches"),
            ("leopard", "leopards"),
            ("rhino", "rhinos"),
            ("elephant", "elephants")
        ]
        self.animal_str_to_idx = {self.animal_names[i][j] : i for i in range(self.n_species) for j in range(2)} # call animal_str_to_idx["zebra"] for 2
        self.animal_idx_to_str = {i : {"singular" : self.animal_names[i][0], "plural" : self.animal_names[i][1]} for i in range(self.n_species)} # call animal_idx_to_str[6]["plural"] for "leopards"

    @property
    def jungle(self):
        '''
        The 'jungle' is everything the current player does not have info on. It is the union of the deck and all the opponent's hands
        '''
        hand_mask = np.array(self.show_hands)[:, None]
        return self.deck + np.sum(self.hands * (1-hand_mask), axis=0)

    @property
    def points_ahead(self):
        '''
        points_ahead counts the points ahead of the strongest opponent
        '''
        return self.scoreboard[self.whose_turn] - max(np.concatenate([self.scoreboard[:self.whose_turn], self.scoreboard[self.whose_turn:]], axis=0))

    @property
    def allowed_actions(self):
        '''
        If it's the AI's turn to play, the allowed actions depend on the hand
        '''
        hand = self.hands[self.whose_turn]
        return [n*util.one_hot(idx, n_dim=self.n_species) for idx in range(len(hand)) for n in range(1, int(hand[idx])+1)]

    @property
    def possible_deck_draws(self):
        '''
        If the AI has a hand of a zebra and two giraffes (represented by self.hands["me_ai"]=[0, 0, 1, 2, 0, 0, 0, 0] ),
        it needs to draw 2 more cards, which could be a meerkat and a rhino (represented by draw=[0, 1, 0, 0, 0, 0, 1, 0])
        This function returns all such possible card draws and their likelihoods

        itertools.combinations_with_replacement(range(3), r=2)
        returns
        iterable((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1) (2, 2))
        '''
        n_hand    = np.sum(self.hands[self.whose_turn, :])
        n_to_draw = min(self.max_n_hand - n_hand, self.n_deck) # restore hand to 5 cards, or draw as many as the deck allows.

        draw_tuples = itertools.combinations_with_replacement(range(self.n_species), r=n_to_draw) # all possible hands that could be drawn, represented as tuples of n_cards rather than arrays of 8
        possible_deck_draws = []
        for draw_tuple in draw_tuples:
            deck_copy = copy.deepcopy(self.deck)
            likelihood = 1.
            for idx in draw_tuple:
                draw_prob = deck_copy / np.sum(deck_copy)
                likelihood *= draw_prob[idx]
                deck_copy -= util.one_hot(idx, n_dim=self.n_species)
            if likelihood > 0.:
                possible_deck_draw = {
                    "for_whom"   : self.whose_turn,
                    "cards"      : sum([util.one_hot(i, n_dim=self.n_species) for i in draw_tuple]),
                    "likelihood" : likelihood
                }
                possible_deck_draws.append(possible_deck_draw)

        return possible_deck_draws

    @property
    def who_next_turn(self):
        return (self.whose_turn + 1) % self.n_players

    @property
    def n_deck(self):
        return np.sum(self.deck)

    @property
    def leading_player(self):
        return np.argmax(self.scoreboard)

    @property
    def is_final(self):
        '''
        The game is over when the deck holds no more cards and one player has emptied his/her hand
        '''
        return np.sum(self.deck) == 0 and (any([np.sum(self.hands[i, :])==0 for i in range(self.n_players)]))

    def apply_action(self, action, verbose=False):
        self.hands[self.whose_turn] -= action # Cards leave the hand of the player
        self.field                  += action # Cards enter the field

        if verbose:
            print(self.player_names[self.whose_turn], " played ", self.animals_arr_to_str(action), "!")

        # Player collects points
        action_animal = np.nonzero(action)[0][0]
        if self.field[action_animal] >= 3:

            # elephants are afraid of mice, elsewise smaller animals are afraid of bigger animals, we use fear_animal=-1 to denote a situation where there's no animals to be chased away
            fear_animal = self.n_species - 1 if action_animal == 0 else max([idx for idx in self.field.nonzero()[0] if idx < action_animal] + [-1])

            if fear_animal >= 0:
                score = self.field[fear_animal]
                self.scoreboard[self.whose_turn] += score # fear_animal is an index denoting which animal it is, but is also related to how many points the animal is worth when chased away
                self.field[fear_animal] = 0

                if verbose:
                    print("The", self.animal_idx_to_str[action_animal]["plural"], "scared away the", self.animal_idx_to_str[fear_animal]["singular" if score == 1 else "plural"], "!")
                    print(self.player_names[self.whose_turn], "scored", score, ("point." if score == 1 else "points."))

        return self

    def apply_deck_draw(self, deck_draw):
        if deck_draw is not None:
            self.hands[deck_draw["for_whom"]] += deck_draw["cards"]
            self.deck -= deck_draw["cards"]

    def next_turn(self):
        self.whose_turn = self.who_next_turn

    def random_deck_draw(self):
        possible_deck_draws = self.possible_deck_draws
        if len(possible_deck_draws) == 0:
            return {"for_whom" : self.whose_turn, "cards" : np.zeros(self.n_species), "likelihood" : 1.}
        else:
            likelihoods = [deck_draw["likelihood"] for deck_draw in possible_deck_draws]
            likelihoods = likelihoods/sum(likelihoods)
            return possible_deck_draws[np.random.choice(range(len(likelihoods)), p=likelihoods)]

    def animals_arr_to_str(self, arr):
        # translate [0, 0, 1, 0, 0, 2, 0, 0] to "2 leopards 1 zebra"
        if np.sum(arr) == 0:
            return "nothing\n"
        else:
            return "\n".join([str(arr[i]) + " " + (self.animal_idx_to_str[i]["singular"] if arr[i] == 1 else self.animal_idx_to_str[i]["plural"]) for i in reversed(range(len(arr))) if arr[i] > 0])

    def action_str_to_arr(self, s):
        try:
            s = s.replace(" ","").lower() # remove whitespace and case-invariant
            if s.isdigit() and len(s) == self.n_species: # if typed like 00030000 for '3 giraffes'
                action = np.array([int(c) for c in s], dtype=int) # change to array
            if "*" in s: # if typed like 3*4 for '3 giraffes'
                n = int(s[0])
                animal_idx = int(s[2])
                action = n*util.one_hot(animal_idx, n_dim=self.n_species)
            else:
                n = int(s[0])
                animal_idx = self.animal_str_to_idx[s[1:]]
                action = n*util.one_hot(animal_idx, n_dim=self.n_species)
        except:
            action = np.zeros(self.n_species)
        return action

    @property # define the print(obj) representation as a single string, so that the subclasses can print it alongside additional info
    def kariba_repstr(self):
        '''
        A repstr (representation string) helps to construct what is printed when we call print on an object of this class
        '''

        s = \
        "********************************\n"+\
        "It is "+str(self.player_names[self.whose_turn])+"\'s turn\n"+\
        "scoreboard:\n"
        for i, player_name in enumerate(self.player_names):
            s += "  "+player_name.ljust(20)+": "+str(self.scoreboard[i])+"\n"
        s += \
        "\nThe deck holds:\n"+\
        util.indent_string(self.animals_arr_to_str(self.jungle), indent_spaces=2)+"\n"+\
        "\nOn the field lies:\n"+\
        util.indent_string(self.animals_arr_to_str(self.field), indent_spaces=2)+"\n"
        for i in range(len(self.show_hands)):
            if self.show_hands[i] == True:
                s += "\n"+str(self.player_names[i])+"\'s hand holds:\n"+\
                util.indent_string(self.animals_arr_to_str(self.hands[i, :]), indent_spaces=2)+"\n"
        s += "--------------------------------\n"
        return s

    def __repr__(self):
        return self.kariba_repstr
