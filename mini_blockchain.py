"""
minimal blockchain:

keep list of transactions in a block. When attempting to add a new transaction, majority has to verify it first. if multiple parties try to hit the same order, process them in chronological order (by timestamp)
But how do we make sure the timestamps are honest? (participants are incentivized to lie about their timestamps). We could do something where the timestamp used is the time at which a certain percentage of the network has seen the order.
    But then wouldn't people be incentivized to lie about other people's timestamps?
    In fact, parties are incentivized to lie about pretty much every aspect of everyone else's transactions...


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
from cryptography.hazmat.primitives import serialization
import socket
import pickle
import datetime
import multiprocessing
import time
from random import randint
"""
requires cryptography library
see https://medium.com/@raul_11817/rsa-with-cryptography-python-library-462b26ce4120
"""

def hash(input_bytes):
    hasher = sha256()
    hasher.update(input_bytes)
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
MESSAGE_DELIMITER = "THISISTHEENDTOKEN".encode()



#TODO: when placing order, add flags for inserting as open order if it's not able to immediately be matched, and whether to match to the extent possible or just to cancel

class OpenOrder:
    """
    Order waiting for hits

    Participant: identifying string for trading party
    Symbol: identifying string for stock
    side: True for selling, False for buying
    price: price desired
    initial_quantity: number of shares desired
    remaining_quantity: number of shares that haven't been fulfilled for this order yet
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


class ExecutedTrade:
    """
    Actual trade occurs when a counterparty hits an open order.
    May not complete the entire order.
    Verification process for transactions should include trade matching to
    automatically convert the appropriate quantity of crossing orders to a trade.
    Should probably wait until end of block, match crossing trades first by price and then by time

    Order: OpenOrder object the trade is hitting
    Counterparty: public identifier string of party hitting the order
    quantity: number of shares counterparty is buying/selling
    timestamp: time at which trade is matched
    """

    def __init__(self, order, counterparty, quantity, timestamp):
        self.order = order
        self.counterparty = counterparty
        self.quantity = quantity
        self.timestamp = timestamp


class MarketState:
    """
    initial_open_orders: dictionary: side -> symbol -> price -> list of open orders. represents state of orderbook at object creation
    initial_quantities_owned: dictionary: party -> symbol -> quantity. represents state of market ownership at object creation
    transaction_list: list of new orders and trades detected since object creation. Note that in practice, clients only issue orders instead of hitting specific trades. Allows us to replay market evolution from beginning
    new_transactions: deque of new orders and executed trades. Sorted by time before processing. used to keep track of executed trades created by crossing orders, destroyed after transaction matching
    current_open_orders: dictionary: side -> symbol -> price -> list of open orders. represents current state of orderbook after match_transactions is called
    current_quantities_owned: dictionary: party -> symbol -> quantity. represents market ownership after match_transactions is called
    current_matched_trades: list of executed trades created by matching orders. populated during match_transactions.
    """
    def __init__(self, initial_open_orders, initial_quantities_owned):
        self.initial_open_orders = initial_open_orders
        self.initial_quantities_owned = initial_quantities_owned
        self.transaction_list = [] # permanent list of new orders and trades (sort by time before matching)
        self.new_transactions = deque()  # time-ordered deque of new orders and hits on those orders (gets destroyed during processing)
        self.current_open_orders = deepcopy(
            initial_open_orders)  # side -> symbol -> price -> time-sorted list of open orders (
        self.current_quantities_owned = deepcopy(
            initial_quantities_owned)  # party -> symbol -> quantity -- after validation and matching
        self.matched_trades = [] # list of executed trade objects.


    def add_transaction(self, transaction):
        self.new_transactions.append(transaction)
        self.transaction_list.append(transaction)


    def match_order(self, order):
        """
        assumes all lists of orders are sorted by time, so it should only be called during match_transactions
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

            relevant_orders = nested_dict_get(self.current_open_orders, (BUY, order.symbol)) # dict of price to list of orders
            if relevant_orders is None:
                if nested_dict_get(self.current_open_orders, (SELL, order.symbol, order.price)) is None:
                    nested_dict_insert(self.current_open_orders, (SELL, order.symbol, order.price), [])

                self.current_open_orders[SELL][order.symbol][order.price].append(order)
                return

            price, order_list = max(relevant_orders.items()) #sorted by first element of tuple, price
            relevant_order = min(order_list, key=lambda rel_order: rel_order.timestamp)

            if relevant_order.price < order.price:
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
            elif funds < order.price * order.remaining_quantity:
                allowed_quantity = funds // order.price
                print(
                    "Failed to place buy order for {0} shares of {1} because you have insufficient funds in your account. Placing order for {2} shares".format(
                        order.quantity, order.symbol, allowed_quantity))
                order.quantity = allowed_quantity

            relevant_orders = nested_dict_get(self.current_open_orders, (SELL, order.symbol))

            if relevant_orders is None:
                if nested_dict_get(self.current_open_orders, (BUY, order.symbol, order.price)) is None:
                    nested_dict_insert(self.current_open_orders, (BUY, order.symbol, order.price), [])

                self.current_open_orders[BUY][order.symbol][order.price].append(order) # TODO: TURN THIS INTO A PRIORITY QUEUE (BY TIME)
                return

            relevant_orders = nested_dict_get(self.current_open_orders, (SELL, order.symbol)) # dict of price to list of orders
            if relevant_orders is None:
                if nested_dict_get(self.current_open_orders, (BUY, order.symbol, order.price)) is None:
                    nested_dict_insert(self.current_open_orders, (BUY, order.symbol, order.price), [])

                self.current_open_orders[BUY][order.symbol][order.price].append(order)
                return

            price, order_list = min(relevant_orders.items()) #sorted by first element of tuple, price
            relevant_order = min(order_list, key=lambda rel_order: rel_order.timestamp)

            if relevant_order.price > order.price:
                if nested_dict_get(self.current_open_orders, (BUY, order.symbol, order.price)) is None:
                    nested_dict_insert(self.current_open_orders, (BUY, order.symbol, order.price), [])
                self.current_open_orders[BUY][order.symbol][order.price].append(order) # TODO: TURN THIS INTO A PRIORITY QUEUE (BY TIME)
                return

            # match with existing open orders
            if relevant_order.remaining_quantity >= order.remaining_quantity:
                matched_trade = ExecutedTrade(relevant_order, order.participant, order.remaining_quantity, order.timestamp)
                self.new_transactions.appendleft(matched_trade)
                return

            else:
                self.new_transactions.appendleft(order)
                matched_trade = ExecutedTrade(relevant_order, order.participant, order.quantity, order.timestamp)
                self.new_transactions.appendleft(matched_trade)
                return

    def process_trade(self, trade):
        """
        process an ExecutedTrade object
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


    # because of the way this is set up (we don't have all the transactions in the block until the block ends) it's probably easiest to implement
    def match_transactions(self):
        """
        match crossing orders and update current state of market
        """
        self.new_transactions = deque(sorted(self.new_transactions, key=lambda transaction: transaction.timestamp))

        for symbol_dict in self.initial_open_orders.values():
            for price_dict in symbol_dict.values():
                for price, order_list in price_dict.items():
                    price_dict[price] = sorted(order_list, key=lambda order: order.timestamp)

        for symbol_dict in self.initial_open_orders.values():
            for price_dict in symbol_dict.values():
                for price, order_list in price_dict.items():
                    price_dict[price] = sorted(order_list, key=lambda order: order.timestamp)

        while len(self.new_transactions) > 0:
            transaction = self.new_transactions.popleft()
            if type(transaction) is OpenOrder:
                self.match_order(transaction)
            elif type(transaction) is ExecutedTrade:
                self.process_trade(transaction)
            else:
                print("invalid transaction type found")  # should log this


class Block:
    """
    index: int. this block's place in the blockchain
    timestamp: datetime. time at which block was found to be valid
    market_state: object representing evolution of orderbook from block initialization to block validation
    previous_hash: hash of previous block.
    nonce: used to change hash of block even when new orders haven't been received
    finder: identifier of client who found valid block
    """

    def __init__(self, index, initial_market_state, previous_block, previous_hash):
        self.index = index
        self.timestamp = None
        self.market_state = initial_market_state
        self.previous_block = previous_block
        self.previous_hash = previous_hash
        self.nonce = 0
        self.finder = None

    def hash_block(self):
        return hash(pickle.dumps(self))

    def add_transaction(self, transaction):
        # transaction can be a new order or a hit on an order
        self.market_state.add_transaction(transaction)

    def generate_next_block(self): # ONLY CALL AFTER SETTING TIMESTAMP AND FINDER
        next_market_state = MarketState(self.market_state.current_open_orders, self.market_state.current_quantities_owned)
        next_block = Block(self.index+1, next_market_state, self, self.hash_block())
        return next_block


# a client has a string identifier, a public/private key pair,
# and an ip address. (I'll use ports here so it can be run locally)
# needs port set (or ip set) in order to know where to announce itself. doesn't need the whole network, just a few access points
class Client:
    """
    Object representing blockchain clients. In order to run the network on a single machine, each client listens to a port, and broadcasts to the ports of other clients it is aware of.
    Clients are run in parallel with multiprocessing.
    identifier: string identifying the client to the public
    port: the port this client will listen on
    port_set: ports of other clients
    private_key: private key for this client. used to sign messages
    public_key: public key for this client. Distributed to other clients, used to authenticate message origin
    public_key_pem: dump of public key, sent in messages since public_key objects can't be pickled
    current_block: block which this client is currently trying to complete
    identifier to public key: dict of identifier->public key for other clients. used to authenticate message origin
    received_messages_set: set of messages received during current block
    received_messages_queue: list of messages received during current block (in order they were received)
    """

    def __init__(self, identifier, port, port_set):
        self.identifier = identifier
        self.port = port # port this listens on.
        self.port_set = port_set # set of active ports
        self.private_key = rsa.generate_private_key(
             public_exponent=65537,
             key_size=2048,
             backend=default_backend()
            )
        self.public_key = self.private_key.public_key()
        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.current_block = None
        self.identifier_to_public_key = {self.identifier: self.public_key}
        self.received_messages_set = set()
        self.received_messages_queue = []

    def receive_announcement(self, announcement_message):
        # currently not in use, but allows clients to handle addition of new nodes to the network even after initial network setup
        message_type, identifier, public_key_pem, port = announcement_message.split(",")
        public_key = serialization.load_pem_private_key(public_key_pem, password=None, backend=default_backend())
        self.port_set.add(port)
        if identifier not in self.identifier_to_public_key:
            self.identifier_to_public_key[identifier] = public_key

    def send_order(self, order):
        print("{} sending order".format(self.identifier))
        self.broadcast(pickle.dumps(("ORDER", order)))

    def receive_order(self, order_message):
        print("{} received order".format(self.identifier))
        message_type, order = pickle.loads(order_message)
        self.current_block.add_transaction(order)

    def receive_order_from_dispatcher(self, order_message):
        print("{} received order from dispatcher".format(self.identifier))
        message_type, order, sender = pickle.loads(order_message)
        if sender == self.identifier:
            self.send_order(order)

    def send_completed_block(self, block):
        print("{} is sending completed block {}".format(self.identifier, block.index))
        self.broadcast(pickle.dumps(("COMPLETE_BLOCK", block)))
        '''for message in self.received_messages_set:
            self.broadcast(pickle.dumps(("BLOCK_MESSAGE", block.index, message)))'''
        self.received_messages_set = set()
        self.received_messages_queue = []

    def receive_completed_block(self, block_message):
        message_type, block  = pickle.loads(block_message)
        print("{} received block candidate {}".format(self.identifier, block.index))
        if block.index >= self.current_block.index:
            self.received_messages_set = set()
            self.received_messages_queue = []
            self.current_block = block.generate_next_block()
            # to verify, replay authenticated messages on block's initial market state and verify that the resulting market state matches that in the given block.
            # also check hash of block and previous block

    def broadcast(self, message): #message must be encoded
        message = message + MESSAGE_DELIMITER
        self.received_messages_queue.append(message)
        self.received_messages_set.add(message)
        for port in self.port_set:
            if port != self.port:
                s = socket.socket()
                host = socket.gethostname()
                s.connect((host, port))
                s.sendall(message)
                s.close()

    def listen(self):
        message_type_to_handler = {"ANNOUNCE": self.receive_announcement, "ORDER":self.receive_order, "ORDER_DISPATCH":self.receive_order_from_dispatcher, "COMPLETE_BLOCK":self.receive_completed_block}
        s = socket.socket()
        host = socket.gethostname()
        s.bind((host, self.port))
        s.listen(5)
        while True:
            # looping over the socket buffer and using a message delimiter allows us to read arbitrary-length messages, although some messages can still be malformed because of pickling errors
            connection, address = s.accept()
            data = connection.recv(2048)
            message_bytes = data
            while data:
                message_bytes += data
                data = connection.recv(2048)
            message_list = [message + '.'.encode() for message in message_bytes.split(MESSAGE_DELIMITER)[:-1]]

            for message in message_list:
                try:
                    # synchronization protocol: broadcast message. keep list of messages seen before. rebroadcast all received messages that haven't been seen before.
                    if message not in self.received_messages_set: # ignore messages you've seen before
                        decoded_message = pickle.loads(message)
                        message_type = decoded_message[0]
                        self.received_messages_set.add(message)
                        self.received_messages_queue.append(message)
                        self.broadcast(message)
                        handler = message_type_to_handler[message_type]
                        handler(message)
                except(pickle.UnpicklingError): # sometimes pickle fails to dump or load large objects properly. We will consider this case a dropped message.
                    print("pickle failure. dropping message")
                except(EOFError):
                    print("pickle failure, dropping message")
                except(UnicodeDecodeError):
                    print("pickle failure, dropping message")
                except(KeyError):
                    print("pickle failure, dropping message")

            self.attempt_block_completion()

    def attempt_block_completion(self):
        self.current_block.finder = self.identifier
        self.current_block.nonce = randint(0, 100000)
        self.current_block.market_state.match_transactions()
        self.current_block.timestamp = datetime.datetime.now()
        dump = pickle.dumps(self.current_block)
        first_hash_char = hash(dump)[0]
        if first_hash_char<10:
            self.send_completed_block(self.current_block)
            # move to next block.
            self.current_block = self.current_block.generate_next_block()

def run_client(client):
    client.listen()

from sas7bdat import SAS7BDAT
d_bids_path = "OrderBook/SampleData/d_bids_sample.sas7bdat"
d_asks_path = "OrderBook/SampleData/d_asks_sample.sas7bdat"
b_bids_path = "OrderBook/SampleData/b_bids_sample.sas7bdat"
b_asks_path = "OrderBook/SampleData/b_bids_sample.sas7bdat"

# in the interest of just getting some trades going, just give every client a massive number of shares of each instrument and a lot of money


def run_system():
    ports = [12345, 12346]
    names = ["first", "second"]
    client_list = []
    for port, name in zip(ports, names):
        client = Client(name, port, set(ports))
        client_list.append(client)

    symbols = set() # sub-sample symbols

    # Can't pickle and unpickle market state when the list of symbols grows too large. subsampling mitigates this somewhat
    with SAS7BDAT(b_bids_path) as b_bids:
        row_idx = 0
        for row in b_bids:
            if row_idx > 0:
                symbol = row[1]
                symbols.add(symbol)
                if len(symbols) > 20: # subsample to keep
                    break
            row_idx += 1
    initial_quantities_owned = {}
    for name in names:
        for symbol in symbols:
            nested_dict_insert(initial_quantities_owned, (name, symbol), 100000)
            nested_dict_insert(initial_quantities_owned, (name, FUNDS), 100000000)

    print("symbols found")

    initial_market_state = MarketState({}, initial_quantities_owned)

    participants_to_public_keys = {}
    for client in client_list:
        participants_to_public_keys[client.identifier] = client.public_key
    genesis_block = Block(0, initial_market_state, None, None)

    for client in client_list:
        client.current_block = genesis_block
        p = multiprocessing.Process(target=run_client, args=(client,))
        p.start()
        time.sleep(1)

    with SAS7BDAT(b_bids_path) as b_bids:
        row_idx = 0

        for row in b_bids:
            if row_idx > 0:
                symbol = row[1]
                if symbol not in symbols:
                    continue
                update_time = row[2]
                side = BUY
                price = row[6]
                shares = row[7]
                executor = client_list[row_idx % len(client_list)]
                order_object = OpenOrder(executor.identifier, symbol, side, price, shares, update_time)
                message = pickle.dumps(("ORDER_DISPATCH", order_object, executor.identifier))
                print("dispatching to {}".format(executor.identifier))
                s = socket.socket()
                host = socket.gethostname()
                s.connect((host, executor.port))
                s.sendall(message + MESSAGE_DELIMITER)
                s.close()

            row_idx +=1

run_system()

'''
p = multiprocessing.Process(target=spawn_client, args=("first", 12345, {12345}))
p.start()
'''

"""
ADD LATER:
DYNAMICALLY ADDING CLIENTS
ASYMETRIC CRYPTOGRAPHY
"""








