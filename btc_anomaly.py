import websocket
import json
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(121)
ay = fig.add_subplot(122)

class AutoEncoder(nn.Module):

    def __init__(self, inputs):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputs, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, inputs)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
    


class SocketFeed(threading.Thread):

    def __init__(self, ticker='BTC-USD'):
        threading.Thread.__init__(self)
        self.url = 'wss://ws-feed.exchange.coinbase.com'
        self.ticker = ticker
        self.bids = {}
        self.asks = {}
        self.sync = False

    def run(self):
        conn = websocket.create_connection(self.url)
        msg = {'type':'subscribe','product_ids':[self.ticker],'channels':['level2_batch']}
        conn.send(json.dumps(msg))
        while True:
            resp = json.loads(conn.recv())
            if 'type' in resp.keys():
                if resp['type'] == 'snapshot':
                    self.bids = {float(i):float(j) for i, j in resp['bids']}
                    self.asks = {float(i):float(j) for i, j in resp['asks']}
                    self.sync = True
                if resp['type'] == 'l2update':
                    for (side, price, volume) in resp['changes']:
                        price, volume = float(price), float(volume)
                        if side == 'buy':
                            if volume == 0:
                                if price in self.bids.keys():
                                    del self.bids[price]
                            else:
                                self.bids[price] = volume
                        else:
                            if volume == 0:
                                if price in self.asks.keys():
                                    del self.asks[price]
                            else:
                                self.asks[price] = volume

    def extraction(self, depth=5):
        bids = list(sorted(self.bids.items(), reverse=True))[:depth]
        asks = list(sorted(self.asks.items()))[:depth]
        bids = np.array(bids)
        asks = np.array(asks)
        bp, bv = bids[:, 0], bids[:, 1]
        ap, av = asks[:, 0], asks[:, 1]
        mid = 0.5*(bp[0] + ap[0])
        bp = (bp - mid) / mid
        ap = (ap - mid) / mid
        bv = bv / sum(bv)
        av = av / sum(av)
        feed = bp.tolist() + ap.tolist() + bv.tolist() + av.tolist()
        return feed, mid



feed = SocketFeed()
feed.start()

epochs = 50
depths = [5, 10, 15, 20, 25, 30, 35, 40]
nnet = {depth:AutoEncoder(depth*4) for depth in depths}
models = {depth:{'Loss':nn.MSELoss(), 'Optimizer':optim.Adam(nnet[depth].parameters()), 'Anomaly':0} for depth in depths}

store_prices = []

while True:
    if feed.sync == True:
        for depth in depths:
            inputs, price = feed.extraction(depth=depth)
            inx = torch.tensor(inputs, dtype=torch.float32)
            models[depth]['Anomaly'] = 0
            for epoch in range(epochs):
                out = nnet[depth](inx)
                loss = models[depth]['Loss'](out, inx)
                models[depth]['Optimizer'].zero_grad()
                loss.backward()
                models[depth]['Optimizer'].step()
                models[depth]['Anomaly'] += loss.item()
        
        store_prices.append(price)
        if len(store_prices) > 40:
            del store_prices[0]
            
        ax.cla()
        ay.cla()
        
        ax.set_title('Anomaly')
        ay.set_title('Bitcoin Price')

        ax.bar(depths, [models[depth]['Anomaly'] for depth in depths], 1.0, 1.0, color='cyan')
        ay.plot(store_prices, color='red')

        plt.pause(0.01)


plt.show()
feed.join()