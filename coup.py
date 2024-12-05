# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Coup implemented in Python.
"""

import enum
import numpy as np
import random
import pyspiel


"""
Coins do not go negative
"""

class Action(enum.IntEnum):
  INCOME = 0
  AID = 1
  COUP = 2
  TAX = 3
  ASSASSINATE = 4
  STEAL = 5
  BLOCKSTEAL = 6
  BLOCKAID = 7
  BLOCKASS = 8
  CHALLENGE = 9
  SWAP1 = 10
  SWAP2 = 11
  FOLD1 = 12
  FOLD2 = 13
  NOCHALL = 14
  DUKE = 15
  CAPTAIN = 16
  CONTESSA = 17
  ASSASSIN = 18


_NUM_PLAYERS = 2

# cap, cont, assassin, duke *3
_DECK = frozenset(["CAPTAIN1", "CAPTAIN2", "CAPTAIN3", "CONTESSA1", "CONTESSA2", "CONTESSA3", 
                   "ASSASSIN1", "ASSASSIN2", "ASSASSIN3", "DUKE1", "DUKE2", "DUKE3"])

_GAME_TYPE = pyspiel.GameType(
    short_name="python_coup",
    long_name="Python Coup",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=len(_DECK),
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=20)  # FIX HERE


class CoupGame(pyspiel.Game):

  def __init__(self, params=None, max_action_count=10, favorite_pl=0):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    self.max_action_count = max_action_count
    self.fav_pl = favorite_pl

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return CoupState(self, max_action=self.max_action_count, favorite_pl=self.fav_pl)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return CoupObserver(pyspiel.IIGObservationType(perfect_recall=True), params)


class CoupState(pyspiel.State):

  def __init__(self, game, max_action=10, favorite_pl=-1):
    """Constructor; should only be called by Game.new_initial_state."""


    super().__init__(game)

    #shuffled_deck = list(_DECK)
    #random.shuffle(shuffled_deck)

    self.action_count = 0
    self.max_action_count = max_action
    self.favorite_pl = favorite_pl

    self.finish_dealing = False
    self.cards = [[], []]
    self.original_cards = []
    self.cards_not_dealt = (len(self.original_cards) < 4)

    # number of influence left
    self.card_numbers = [2,2]

    # when a player need a new card
    self.swap = -1

    self.coins = [2,2]

    # can only challenge or not
    self.challenge_round = [False, False]

    # does the player has a pending assassination
    self.ass_att = [False, False]

    # forced to fold
    self.fold = [False, False]

    # prev action done by the player
    self.PrevClaimedCharacters = ['','']
    self.prevAction = [Action.INCOME, Action.INCOME]

    self._game_over = (0 in self.card_numbers)

    # always start with player 0
    self._next_player = 0


  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""


    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self.swap >= 0 or (self.cards_not_dealt and not self.finish_dealing) :
      return pyspiel.PlayerId.CHANCE
    else:
      return self._next_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0

    cards = self.cards[player]

    # challenged round
    if self.challenge_round[player]:
      self.challenge_round[player] = False
      return [Action.CHALLENGE, Action.NOCHALL]
    
    # result in a kill
    
    if self.fold[player] or self.ass_att[1-player]:
      actions = []
      if cards[0] != '':
        actions.append(Action.FOLD1)
      if cards[1] != '':
        actions.append(Action.FOLD2)
      
      if len(actions) == 0:
        self._game_over = True
        self.card_numbers[player] = 0
        return [Action.INCOME]
      
      actions.sort()      
      return actions
      
    # normal action round
    if self.coins[self._next_player] >= 10:
      return [Action.COUP]

    actions = [Action.AID, Action.INCOME, Action.TAX]
    # skills = [Action.AID, Action.TAX, Action.ASSASSINATE, Action.STEAL, Action.BLOCKAID, Action.BLOCKASS, Action.BLOCKSTEAL]
    if self.coins[1 - self._next_player] > 0:
      actions += [Action.STEAL]

    if self.coins[self._next_player] >= 3:
      actions += [Action.ASSASSINATE]
    
    if self.coins[self._next_player] >= 7:
      actions += [Action.COUP]

    if self.prevAction[1 - self._next_player] == Action.AID:
      actions += [Action.BLOCKAID]

    if self.prevAction[1 - self._next_player] == Action.STEAL:
      actions += [Action.BLOCKSTEAL]

    if self.prevAction[1 - self._next_player] == Action.ASSASSINATE:
      actions += [Action.BLOCKASS]

    actions.sort()
    return actions

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = sorted(_DECK - set(self.cards[0]) - set(self.cards[0]))
    p = 1.0 / len(outcomes)
    # trim the suffix
    res = [(Action[o[:-1]], p) for o in outcomes]
    return res

  def _apply_action(self, action):

    self.action_count += 1

    if self.action_count > self.max_action_count:
      # early abort
      self._game_over = True
      return

    """Applies the specified action to the state."""
    if self.is_chance_node():
      char = Action._value2member_map_[action].name


      if self.swap == 0:
        self.swap = -1
        if self.prevAction[0] == Action.SWAP1:
          self.cards[0][0] = char
        else:
          self.cards[0][1] = char
        return
      
      elif self.swap == 1:
        self.swap = -1

        if self.prevAction[1] == Action.SWAP1:
          self.cards[1][0] = char
        else:
          self.cards[1][1] = char
      
        return
    
      if self.cards_not_dealt and (not self.finish_dealing):
        self.original_cards.append(char)
        self.cards_not_dealt = (len(self.original_cards) < 4)
        if not self.cards_not_dealt:
          # append the cards
          self.cards[0] = self.original_cards[:2]
          self.cards[1] = self.original_cards[2:]
          self.finish_dealing = True
      return



    # normal action node
    else:
      self.prevAction[self._next_player] = action

      #  need to fold
      if action == Action.FOLD1:
        self.cards[self._next_player][0] = ''
        
        self.card_numbers[self._next_player] -= 1
        
        self.ass_att[1-self._next_player] = False
        self.fold[self._next_player] = False


        if self.card_numbers[self._next_player] == 0:
            self._game_over = True
            return

        self._next_player = 1- self._next_player

      # need to fold
      elif action == Action.FOLD2:
        self.cards[self._next_player][1] = ''
        self.card_numbers[self._next_player] -= 1

        self.ass_att[1-self._next_player] = False
        self.fold[self._next_player] = False
        
        if self.card_numbers[self._next_player] == 0:
            self._game_over = True
            return

        self._next_player = 1- self._next_player



      elif action == Action.INCOME:
        self.coins[self._next_player] += 1
        self._next_player = 1 - self._next_player
      
      elif action == Action.AID:
        self.coins[self._next_player] += 2
        self._next_player = 1- self._next_player
      
      elif action == Action.TAX:
        self.coins[self._next_player] += 3
        self.PrevClaimedCharacters[self._next_player] = "DUKE"
        self.challenge_round[1-self._next_player] = True
        self._next_player = 1- self._next_player
      
      # other actions
      elif action == Action.COUP:
        self.coins[self._next_player] -= 7
        if self.card_numbers[1-self._next_player] == 1:
          # won
          self.card_numbers[1-self._next_player] = 0
          self._game_over = True
          return
        
        self.fold[1-self._next_player] = True
        self._next_player = 1- self._next_player


      elif action == Action.ASSASSINATE:
        self.coins[self._next_player] -= 3
        self.PrevClaimedCharacters[self._next_player] = "ASSASSIN"
        self.challenge_round[1-self._next_player] = True
        self.ass_att[self._next_player] = True
        self._next_player = 1- self._next_player


      elif action == Action.STEAL:
        self.coins[self._next_player] += 2
        self.coins[1 - self._next_player] -= 2

        self.PrevClaimedCharacters[self._next_player] = "CAPTAIN"
        self.challenge_round[1-self._next_player] = True
        self._next_player = 1- self._next_player

      elif action == Action.BLOCKAID:
        self.coins[1 - self._next_player] -= 2
        self.PrevClaimedCharacters[self._next_player] = "DUKE"
        self.challenge_round[1-self._next_player] = True
        self._next_player = 1- self._next_player

      
      elif action == Action.BLOCKSTEAL:
        self.coins[self._next_player] += 2
        self.coins[1 - self._next_player] -= 2
        self.PrevClaimedCharacters[self._next_player] = "CAPTAIN"
        self.challenge_round[1-self._next_player] = True

        self._next_player = 1- self._next_player
      

      elif action == Action.BLOCKASS:
        
        self.PrevClaimedCharacters[self._next_player] = "CONTESSA"
        self.ass_att[self._next_player] = False
        self.challenge_round[1-self._next_player] = True

        self._next_player = 1- self._next_player

      elif action == Action.NOCHALL:

        return

      elif action == Action.CHALLENGE:
      
        op = 1 - self._next_player
        cards = self.cards[op]
        char = self.PrevClaimedCharacters[op]
        solid = self.PrevClaimedCharacters[op] in cards

        # print("BLUFF CHECK", cards, self.PrevClaimedCharacters[op])
        # print(solid)
        # not bluffing

        if solid:

          # special case
          if self.prevAction[op] == Action.ASSASSINATE:
            # deadly:
            if "CONTESSA" in self.cards[1-op]:
              self.fold[1-op] = True
              
            else:
                self.card_numbers[1-op] = 0
                self._game_over = True
                return
            
          # you need to fold
          self.fold[1-op] = True
          
          if (cards[0] == char):
            self.prevAction[op] = Action.SWAP1
          else:
            self.prevAction[op] = Action.SWAP2

          self.swap = op
          self._next_player = 1 - op

        # bluffing
        # 1. revert action
        # 2. op needs to fold
        else:
          prev_action = self.prevAction[op]
          if prev_action == Action.TAX:
            self.coins[op] -= 3
          if prev_action == Action.ASSASSINATE:
            self.coins[op] += 3
            self.ass_att[op] = False
          if prev_action == Action.BLOCKAID:
            self.coins[1-op] += 2
          if prev_action == Action.BLOCKASS:
            # deadly
            self.card_numbers[op] = 0
            self._game_over = True
            return
          if prev_action == Action.BLOCKSTEAL:
            self.coins[self._next_player] += 2
            self.coins[op] -= 2

          if prev_action == Action.STEAL:
            self.coins[self._next_player] += 2
            self.coins[1 - self._next_player] -= 2
          
          self.fold[op] = True
          self._next_player = op
      
      self._game_over = (self.card_numbers[0] <= 0) or (self.card_numbers[1] <= 0)
      return


  def _action_to_string(self, player, action):
    return ''


  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over


  def returns(self):
    # game broke early
    if self.action_count > self.max_action_count:
      if self.favorite_pl == 0:
        return [1, -1]
      elif self.favorite_pl == 1:
        return [-1, 1]
      else:
        if self.card_numbers[0] > self.card_numbers[1]:
            return [1, -1]
        if self.card_numbers[0] < self.card_numbers[1]:
            return [-1, 1]
        if self.coins[0] > self.coins[1]:
            return [1, -1]
        if self.coins[0] < self.coins[1]:
            return [-1, 1]
        
        return [-1, 1]
    # normal end
    else:
      if self.card_numbers[0] == 0:
        return [-1, 1]
      else:
        return [1, -1]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return "".join([str(c) for c in self.cards])


class CoupObserver:
    """Simplified Observer for Coup with dict attribute."""

    def __init__(self, iig_obs_type, params=None):
        """Initializes an empty observation tensor and dict."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")

        self.private_info = iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER
        self.public_info = iig_obs_type.public_info

        # Observation components
        pieces = [("player", 2, (2,)),  # Current player indicator
                  ("private_cards", 4, (4,)),  # Current player's private cards
                  ("coins", 2, (2,)), # Coin counts for both players
                  ("p0_prevmove", 14, (14, )), # self prev move
                  ("p1_prevmove", 14, (14, ))]# self prev move

        # Build the single flat tensor
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, dtype=np.float32)

        # Build the named views of the tensor
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index:index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        """Updates the tensor and dict to reflect `state` from the POV of `player`."""
        self.tensor.fill(0)  # Reset the tensor

        # Set current player information
        self.dict["player"][player] = 1

        # Set private cards
        card_list = ["CAPTAIN", "CONTESSA", "ASSASSIN", "DUKE" '']
        if self.private_info:
            for card in state.cards[player]:
                if card in card_list:
                    if len(card) > 0:
                      card = card[:-1]
                    card_index = card_list.index(card)
                    self.dict["private_cards"][card_index] = 1

        # Set public information
        move_idx0 = state.prevAction[0]
        move_idx1 = state.prevAction[1]

        if self.public_info:
            self.dict["coins"][:] = state.coins
            self.dict['p0_prevmove'][move_idx0] = 1
            self.dict['p1_prevmove'][move_idx1] = 1
            #self.dict['round_numer'][0] = state.action_count

    def string_from(self, state, player):
        """Returns a simple string representation of the observation."""
        return f"Player {player} | Cards: {state.cards[player]} | Coins: {state.coins}"
    
# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, CoupGame)
