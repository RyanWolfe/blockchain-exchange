"""
minimal blockchain:

proof of work: necessary to provide security.
Ensure that all valid transactions are included, no invalid changes are.

keep list of transactions in a block. When attempting to add a new transaction, majority has to verify it first. if multiple parties try to hit the same order, process them in chronological order (by timestamp)
But how do we make sure the timestamps are honest? (participants are incentivized to lie about their timestamps). We could do something where the timestamp used is the time at which a certain percentage of the network has seen the order.
    But then wouldn't people be incentivized to lie about other people's timestamps?
    In fact, parties are incentivized to lie about pretty much every aspect of everyone else's transactions...

    look at how cash to crypto exchanges work.


what if we just had instantaneous addition of a block, with different measures to avoid changes?

In the case of stocks, tokens can represent shares.
different token for each symbol, or for each share?

derivatives could be implemented as smart contracts acting on relevant tokens

block data: list of transactions.


Problems:
incentivizing block creation
what if two parties try to hit the same bid? who gets it?

could have regulators force a standardized protocol...


consider using smart contracts to enforce order. look into ethereum more.
want currency in the chain so monetary transactions are tied to trades?
but I would guess traders wouldn't like being forced into crypto exposure in order to execute normal trades. What if it's backed by regular currency? Network has a bank account, only way to generate network currency is paying into that account, network currency can be spent by withdrawing from the account? that would require a central party for actually retrieving money, but records, verification, regulatory stuff could still be decentralized.

problem: in bitcoin, most people don't have much incentive to lie about other people's transactions. Most of the design is focused on preventing a small number of people from lying about transactions, and preventing people from changing transactions by changing a past block. in a distributed market, people have an incentive to lie about transactions they're not involved in.


"""

from hashlib import sha256
from copy import copy, deepcopy
from collections import deque
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from  cryptography.hazmat.primitives import serialization
import socket

"""
requires cryptography library
see https://medium.com/@raul_11817/rsa-with-cryptography-python-library-462b26ce4120
"""


def hash(input_string):
    hasher = sha256()
    hasher.update(input_string)
    return hasher.digest()


def nested_dict_get(nested_dict, key_tuple):
    """
    Access a value in a multilevel dictionary
    Returns None if keys are not all in dictionary
    Returns value if keys are all in dictionary
    """
    for key in key_tuple:
        if type(nested_dict) != dict:
            raise ValueError("Overspecified key: attempted to index into a non-dictionary object")
        if key not in nested_dict:
            return None
        nested_dict = nested_dict[key]
    return nested_dict  # after looping through the key tuple, this will in fact be the value we wanted.


def nested_dict_insert(nested_dict, key_tuple, value):
    for key in key_tuple[:-1]:
        if type(nested_dict) != dict:  # there's already a value here, and it's not a dictionary
            raise ValueError("Overspecified key: attempted to index into a non-dictionary object")
        if key not in nested_dict:
            nested_dict[key] = {}
        nested_dict = nested_dict[key]
    final_key = key_tuple[-1]
    nested_dict[final_key] = value


SELL = True
BUY = False
FUNDS = 'FUNDS'


#TODO: when placing order, add flags for inserting as open order if it's not able to immediately be matched, and whether to match to the extent possible or just to cancel

class OpenOrder:
    """
    Order waiting for hits

    Participant: identifying string for trading party
    Symbol: identifying string for stock
    side: True for selling, False for buying
    Quantity: number of stock units desired
    Timestamp: Time order is created

    """

    def __init__(self, participant, symbol, side, price, quantity, timestamp):
        self.participant = participant
        self.symbol = symbol
        self.side = side
        self.price = price
        self.initial_quantity = quantity
        self.remaining_quantity = quantity
        self.timestamp = timestamp

    def __str__(self):
        return str(self.participant) + str(self.symbol) + str(self.price) + str(self.initial_quantity) + str(self.remaining_quantity) + str(self.timestamp)


class ExecutedTrade:
    """
    Actual trade occurs when a counterparty hits an open order.
    May not complete the entire order.
    Verification process for transactions should include trade matching to
    automatically convert the appropriate quantity of crossing orders to a trade.
    Should probably wait until end of block, match crossing trades first by price and then by time

    Order: OpenOrder object the trade is hitting
    Counterparty:

    """

    def __init__(self, order, counterparty, quantity, timestamp):
        self.order = order
        self.counterparty = counterparty
        self.quantity = quantity
        self.timestamp = timestamp

    def __str__(self):
        return str(self.order) + str(self.counterparty) + str(self.quantity) + str(self.timestamp)



"""
There are two ways to execute a trade: Either hit an open order explicitly, or place your own order, and if it causes the book to cross it will be fulfilled to the extent possible.

What I'm going to do for now: clients will only put in  orders, because all the functionality of trades is embedded in orders.
Also I probably don't actually want clients to be able to choose between different 
"""

"""
Consider:
build market model from scratch? implement differently?

Maybe use case is better for broker-type (high volume, low frequency) trades than for high frequency?

or base on current model, relax one assumption at a time.
"""


class MarketState:
    """
    dictionary side -> symbol -> price -> time-sorted list of open orders

    should probably also keep track of quantity owned, so we know whether sellers actually can sell.
    party -> symbol -> quantity owned

    If we use an in-market currency, we can also check whether a buyer has enough money to buy.
    """

    def __init__(self, initial_open_orders, initial_quantities_owned):
        self.initial_open_orders = initial_open_orders  # side -> symbol -> price -> list of open orders
        self.initial_quantities_owned = initial_quantities_owned  # party -> symbol -> quantity
        self.transaction_list = [] # list of new orders and trades (sort by time before matching)
        self.new_transactions = deque()  # time-ordered deque of new orders and hits on those orders
        self.current_open_orders = deepcopy(
            initial_open_orders)  # side -> symbol -> price -> time-sorted list of open orders (
        self.current_quantities_owned = deepcopy(
            initial_quantities_owned)  # party -> symbol -> quantity -- after validation and matching
        self.matched_trades = [] # list of executed trade objects.


    def __str__(self):
        pass

    def add_transaction(self, transaction):
        # validate that it comes from sender
        self.new_transactions.append(transaction)
        self.transaction_list.append(transaction) #sort when matching trades


    def match_order(self, order):
        """
        SHOULD ONLY BE CALLED DURING match_transactions due to the assumptions it makes
        """
        if order.side == SELL:
            quantity_owned = nested_dict_get(self.current_quantities_owned,
                                             (order.participant, order.symbol))
            if quantity_owned is None:
                print("Failed to place sell order for {} shares of {} because you own no shares".format(
                    order.quantity, order.symbol))
            elif quantity_owned < order.initial_quantity:
                print(
                    "Failed to place sell order for {0} shares of {1} because you only own {0}. Placing sell order for {0} shares".format(
                        order.initial_quantity, order.symbol))
                order.initial_quantity = quantity_owned
                order.remaining_quantity = order.initial_quantity
            # match against open buy orders
            relevant_orders = sorted(self.current_open_orders[BUY][order.symbol].items(), reverse=True)
            price, relevant_order = relevant_orders.items()[0]
            if price < order.price:
                #place new order
                if nested_dict_get(self.current_open_orders, (SELL, order.symbol, order.price)) is None:
                    nested_dict_insert(self.current_open_orders, (SELL, order.symbol, order.price), [])

                self.current_open_orders[SELL][order.symbol][order.price].append(order)
                return

            # match with existing open orders
            if relevant_order.quantity >= order.quantity: #convert this open order to a trade.
                matched_trade = ExecutedTrade(relevant_order, order.participant, order.quantity, order.timestamp)
                self.new_transactions.appendleft(matched_trade)
                return

            else:  # re-prepend the remainder of the new order, then prepend the matched trade
                self.new_transactions.appendleft(order)

                matched_trade = ExecutedTrade(relevant_order, order.participant, order.quantity, order.timestamp)
                self.new_transactions.appendleft(matched_trade)
                return

        if order.side == BUY:
            funds = nested_dict_get(self.current_quantities_owned, (order.participant, FUNDS))
            if funds is None:
                print("Failed to place buy order for {} shares of {} because you have no funds in your account".format(
                    order.quantity, order.symbol))
            elif funds < order.price * order.quantity:
                allowed_quantity = funds // order.price
                print(
                    "Failed to place buy order for {0} shares of {1} because you have insufficient funds in your account. Placing order for {2} shares".format(
                        order.quantity, order.symbol, allowed_quantity))
                order.quantity = allowed_quantity

            relevant_orders = sorted(self.current_open_orders[SELL][order.symbol].items())
            price, existing_order = relevant_orders.items()[0]

            if price > order.price:
                if nested_dict_get(self.current_open_orders, (BUY, order.symbol, order.price)) is None:
                    nested_dict_insert(self.current_open_orders, (BUY, order.symbol, order.price), [])

                self.current_open_orders[BUY][order.symbol][order.price].append(order) # TODO: TURN THIS INTO A PRIORITY QUEUE (BY TIME)
                return

            # match with existing open orders
            if existing_order.quantity >= order.quantity:
                matched_trade = ExecutedTrade(order, order.participant, order.quantity, order.timestamp)
                self.new_transactions.appendleft(matched_trade)
                return

            else:
                self.new_transactions.appendleft(order)

                matched_trade = ExecutedTrade(order, order.participant, order.quantity, order.timestamp)
                self.new_transactions.appendleft(matched_trade)
                return

    def process_trade(self, trade):
        """
        should only be called during match_transactions
        """
        #make sure referenced order is still open and large enough
        referenced_order = trade.order

        if trade.order.remaining_quantity < trade.quantity:
            #log error
            print("Trade size larger than order size. cancelling trade.")

        trade.order.remaining_quantity -= trade.quantity

        if trade.order.remaining_quantity == 0: # close the order by removing from list of open orders.
            relevant_orders = self.current_open_orders[trade.order.side][trade.order.symbol][trade.order.price]
            self.current_open_orders[trade.order.side][trade.order.symbol][trade.order.price] = [relevant_order for relevant_order in relevant_orders if relevant_order.remaining_quantity >0 ]

        self.matched_trades.append(trade)


    #because of the way this is set up (we don't have all the transactions in the block until the block ends) it's probably easiest to implement

    def match_transactions(self):
        # go through lists of pending trades/orders, match off.
        self.new_transactions = sorted(self.new_transactions, key=lambda transaction: transaction.timestamp)
        self.transaction_list = sorted(self.transaction_list, key=lambda transaction: transaction.timestamp)

        # side -> symbol -> price -> time-sorted list of open orders
        for symbol_dict in self.initial_open_orders.values():
            for price_dict in symbol_dict.values:
                for price, order_list in price_dict.items():
                    price_dict[price] = sorted(order_list, key=lambda order: order.timestamp)

        for symbol_dict in self.initial_open_orders.values():
            for price_dict in symbol_dict.values:
                for price, order_list in price_dict.items():
                    price_dict[price] = sorted(order_list, key=lambda order: order.timestamp)

        while len(self.new_transactions) > 0:
            transaction = self.new_transactions.popleft()
            if type(transaction) is OpenOrder:
                self.match_order(transaction)

            elif type(transaction) is ExecutedTrade:
                self.process_trade(transaction)
                # check that referenced order is still open
                # check that the issuer has the resources to execute this trade
                # if it is, fill it to the extent that it's valid.
                # if not completely filled, check for other open orders that meet the criteria
                # TODO: add option to convert unfulfilled portion to open order
            else:
                print("invalid transaction type found")  # should log this


"""
For now I'll do trade matching when a block is placed on the chain, in order to give time to reach consensus about ordering. May change later to allow for faster updates to market state.
"""


class Block:
    """
    index: int. this block's place in the blockchain
    timestamp: datetime. time of insertion into chain.
    transaction_list: list(OpenOrder || Trade) -- a time-ordered series of transactions
    previous_hash: hash of previous block.
    """

    def __init__(self, index, timestamp, initial_market_state, transaction_list, participant_identifier_to_public_key, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.initial_market_state = initial_market_state
        self.transaction_list = transaction_list  # time-ordered list
        self.participant_identifier_to_public_key = participant_identifier_to_public_key
        self.previous_hash = previous_hash
        self.hash = self.hash_block()

    def hash_block(self):
        return hash(str(self.index) +
                    str(self.timestamp) +
                    str(self.initial_market_state) +
                    str([participant_identifier + str(public_key) for participant_identifier, public_key in self.participant_identifier_to_public_key.items()]) +
                    str([str(transaction) for transaction in self.transaction_list]) +
                    str(self.previous_hash))

    def add_transaction(self, transaction):
        # transaction can be a new order or a hit on an order
        self.transaction_list.append(transaction)


    def match_and_validate(self):
        pass
    # close fulfilled orders
    # match trades (by price, then time)
    # figure out which orders are still open
    def validate_transaction_list(self):
        open_orders = copy(self.initial_open_orders)
        for transaction in self.transaction_list:
            if type(transaction) is OpenOrder:
                pass


# a client has a string identifier, a public/private key pair,
# and an ip address. (I'll use ports here so it can be run locally)
# needs port set (or ip set) in order to know where to announce itself. doesn't need the whole network, just a few access points
class Client:

    def __init__(self, identifier, port,  port_set):
        self.identifier = identifier
        self.private_key = rsa.generate_private_key(
             public_exponent=65537,
             key_size=2048,
             backend=default_backend()
            )
        self.public_key = self.private_key.public_key()

        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
            encryption_algorithm=serialization.NoEncryption
        )

        self.port = port # port this listens on.
        self.port_set = set() # set of active ports

        self.announce()
        # announce

    def announce(self):
        # broadcast identifier, public key, listening port
        self.broadcast("ANNOUNCE,{},{},{}".FORMAT(self.identifier, self.public_key_pem, self.port))


    def receive_announcement(self, announcement_message):
        pass

    def send_order(self, order_message):
        #broadcast order
        pass

    def process_order(self, order_message):
        pass

    def broadcast_block_end(self):
        pass

    def check_block_end(self):
        pass

    def broadcast(self, message):
        for port in self.port_set:
            s = socket.socket()
            host = socket.gethostname()
            s.connect((host, port))
            s.sendall(message.encode())
            s.close()


    def listen(self):

        message_type_to_handler = {"ANNOUNCE": self.receive_announcement, }

        s = socket.socket()
        host = socket.gethostname()
        port = 22345
        s.bind((host, port))

        s.listen(5)
        while True:
            connection, address = s.accept()
            print("got connection from " + address[0] + str(address[1]))

            data = connection.recv(4096)
            message = data.decode()
            while data:
                message += data.decode()
            message_type = data.split(",")[1]

            handler = message_type_to_handler[message]
            handler(message)


            connection.send("connection successful")
            connection.close()




    #should broadcast public key to other nodes, then broadcast signed orders





