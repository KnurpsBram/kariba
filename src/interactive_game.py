import time
import copy
import numpy as np
import IPython.display as ipd

import kariba_moismcts
import util

class InteractiveKaribaGame():
    def __init__(self, kariba, show_deck, show_opponent_hand, n=500, indent_spaces=4):
        self.kariba = kariba
        self.human_name = kariba.player_names[0]
        self.ai_name = kariba.player_names[1]

        self.n = n

        self.show_deck = show_deck
        self.show_opponent_hand = show_opponent_hand
        self.indent_spaces = indent_spaces

        self.n_species = self.kariba.n_species
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

        self.text_emojis_bad = [
            "(╯°□°）╯︵ ┻━┻",
            "༼ つ ಥ_ಥ ༽つ",
            "ಠ_ಠ",
            "(ノಠ益ಠ)ノ彡┻━┻",
            "ಠ╭╮ಠ",
            "ლ(ಠ益ಠლ)",
            "┻━┻ ︵ヽ(`Д´)ﾉ︵ ┻━┻",
            "(ಥ﹏ಥ)",
            "(°ロ°)",
            "(ʘᗩʘ')"
        ]
        self.text_emojis_good = [
            "( ͡° ͜ʖ ͡°)",
            "¯\_(ツ)_/¯",
            "(づ｡◕‿‿◕｡)づ",
            "(☞ﾟ∀ﾟ)☞",
            "ヾ(⌐■_■)ノ♪",
            "~(˘▾˘~)",
            "(~˘▾˘)~",
            "(•_•) ( •_•)>⌐■-■ (⌐■_■)",
            "☜(⌒▽⌒)☞"
        ]

    def animals_arr_to_str(self, arr):
        # translate [0, 0, 1, 0, 0, 2, 0, 0] to "2 leopards 1 zebra"
        if np.sum(arr) == 0:
            return "nothing"
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
                animal_idx = [i for i in range(self.n_species) if any([word in s for word in self.animal_names[i]])]
                if len(animal_idx) >= 2:
                    print("you submitted more than 1 animal, we'll ignore that and just select the first")
                animal_idx = animal_idx[0]
                n = [int(c) for c in s if c.isdigit()]
                if len(n) > 1:
                    print("you submitted more than 1 number, we'll ignore that and just select the first")
                if len(n) == 0:
                    print("you submitted no number. We'll assume you meant to play a single card")
                    n = [1]
                n = n[0]
                action = n*util.one_hot(animal_idx, n_dim=self.n_species)
        except:
            action = np.zeros(self.n_species)
        return action

    def get_action_from_human(self):
        time.sleep(1)
        cards = self.action_str_to_arr(input("What's your move?"))
        if any([np.array_equal(cards, allowed_action["cards"]) for allowed_action in self.kariba.allowed_actions(self.human_name)]):
            event = {
                "kind"  : "action",
                "who"   : self.human_name,
                "cards" : cards
            }
            return event
        else:
            print("Erm. That's not a valid move")
            return self.get_action_from_human() # recursion!

    def show_state(self):
        print("********************************************************")
        print("Scoreboard:")
        print(util.indent_string("\n".join([player.ljust(max([len(name) for name in self.kariba.player_names]))+" : "+str(score) for player, score in self.kariba.scoreboard.items()]), indent_spaces=self.indent_spaces))
        print("")

        if self.show_deck and self.show_opponent_hand:
            print("The deck holds:")
            print(util.indent_string(self.animals_arr_to_str(self.kariba.deck), indent_spaces=self.indent_spaces))
        elif self.show_deck and not self.show_opponent_hand:
            print("The 'jungle' (the jungle is the union of the deck and the opponent's hand) holds:")
            print(util.indent_string(self.animals_arr_to_str(self.kariba.jungle(self.human_name)), indent_spaces=self.indent_spaces))
        print("")

        print("On the field lies:")
        print(util.indent_string(self.animals_arr_to_str(self.kariba.field), indent_spaces=self.indent_spaces))
        print("")

        if self.show_opponent_hand:
            print(self.ai_name+"'s hand holds:")
            print(util.indent_string(self.animals_arr_to_str(self.kariba.hands[self.ai_name]), indent_spaces=self.indent_spaces))
            print("")

        print(self.human_name+"'s hand holds:")
        print(util.indent_string(self.animals_arr_to_str(self.kariba.hands[self.human_name]), indent_spaces=self.indent_spaces))
        print("")

    def process_event(self, event):

        print("--------------------------------------------------------")

        state_before_event = copy.deepcopy(self.kariba)
        self.kariba.apply_event(event)

        if event["kind"] == "deck_draw":
            if event["who"] == self.human_name:
                print(self.human_name, "drew the following card"+("s" if np.sum(event["cards"])>1 else "")+":")
                print(util.indent_string(self.animals_arr_to_str(event["cards"]), indent_spaces=self.indent_spaces))
            if event["who"] == self.ai_name:
                if self.show_opponent_hand:
                    print(self.ai_name, "drew the following card"+("s" if np.sum(event["cards"])>1 else "")+":")
                    print(util.indent_string(self.animals_arr_to_str(event["cards"]), indent_spaces=self.indent_spaces))
                else:
                    print(self.ai_name, "drew new cards.")

        if event["kind"] == "action":
            print(event["who"], "played:")
            print(util.indent_string(self.animals_arr_to_str(event["cards"]), indent_spaces=self.indent_spaces))
            score_gained = self.kariba.scoreboard[event["who"]] - state_before_event.scoreboard[event["who"]]
            if score_gained > 0:
                change = state_before_event.field - self.kariba.field
                chaser = (change<0)*self.kariba.field
                chasee = (change>0)*state_before_event.field

                print("")
                print("!!!")
                print( "The", self.animals_arr_to_str(chaser).replace("\n", ""), "chased away the ", self.animals_arr_to_str(chasee).replace("\n", ""), "!" )
                print( event["who"]+" scored "+str(score_gained)+" point"+("s" if score_gained > 1 else ""), np.random.choice(self.text_emojis_bad) if event["who"] == self.ai_name else np.random.choice(self.text_emojis_good))
                print("!!!")

        print("")

    def play_game(self):
        while not self.kariba.is_final:
            random_card_draw = self.kariba.random_card_draw()
            self.process_event(random_card_draw)

            if self.kariba.whose_turn == self.human_name:
                self.show_state()
                action = self.get_action_from_human()
                # ipd.clear_output()

            if self.kariba.whose_turn == self.ai_name:
                print(self.ai_name, "is planning its next move...")
                time.sleep(1)
                action = kariba_moismcts.moismcts(copy.deepcopy(self.kariba), n=self.n)

            self.process_event(action)
            self.kariba.next_turn()

        self.show_state()
        print(self.kariba.leading_player, " won!")

def interactive_game(n=500):
    human_name         = input("Okay Human! what is your name? ")
    show_opponent_hand = util.str_to_bool(input("Do you want the AI's cards to be visible to you? (y/n)"))
    show_deck          = util.str_to_bool(input("Do you want the contents of the deck to be visible to you? (y/n)"))

    ai_name      = "Monty Carlos"
    player_names = [human_name, ai_name]

    print("Okay, let's flip a coin to see who may begin the game")
    whose_turn_ = np.random.randint(2)
    print("Very well! ", player_names[whose_turn_], " may begin! \n")

    interactive_game = InteractiveKaribaGame(kariba_moismcts.Kariba(player_names = player_names, whose_turn_ = whose_turn_), show_deck, show_opponent_hand, n=n)

    interactive_game.play_game()
