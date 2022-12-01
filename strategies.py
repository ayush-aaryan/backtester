import numpy as np
import math


class TrendFollowingStrategy(object):
    def __init__(self, settings):
        self.settings = {
            'contract_type': 'long', 
            'risk_factor': 0.000,
            'fast_window': 12,
            'slow_window': 26,
            'signal_window': 9,
        }
        if type(settings) == dict:
            if all(key in settings for key in self.settings.keys()):
                self.settings = settings
        self.long_on = True if self.settings['contract_type'] in ['long', 'both'] else False
        self.short_on = True if self.settings['contract_type'] in ['short', 'both'] else False
        self.risk_factor = self.settings['risk_factor']
        self.fast_window = self.settings['fast_window']
        self.slow_window = self.settings['slow_window']
        # self.std_window = self.settings['std_window']
        self.signal_window = self.settings['signal_window']
        # self.exit_multiplier = self.settings['exit_multiplier']
        self.per_trade = 10000
        self.required_data_length = 3 * np.max([
            self.fast_window,
            self.slow_window,
            self.signal_window
        ])
        return

    def __str__(self):
        description = "Future Trend Following Strategy"
        description += "\nintroduced by Andreas Clenow in his Trading Evolved book"
        return description

    def position_size(self, ohlc, portfolio_value):
        size = math.floor(self.per_trade/ohlc["close"].iloc[-1])
        return size

    def compute_indicators(self, ohlc):
        print(ohlc.shape[0])
        if ohlc.shape[0] < self.required_data_length:
            raise Exception("Length of passed OHLC data too small")
        ohlc['ema_fast'] = ohlc.ewm(self.fast_window).mean().close.values
        ohlc['ema_slow'] = ohlc.ewm(self.slow_window).mean().close.values
        ohlc['macd_line'] = ohlc.ewm(self.fast_window).mean().close.values - ohlc.ewm(self.slow_window).mean().close.values
        ohlc['signal_line'] = ohlc.ewm(self.signal_window).mean().macd_line.values
        ohlc['uptrend'] = np.where((ohlc['ema_fast'] >= ohlc['ema_slow']) & (ohlc['macd_line']> ohlc['signal_line']) , True, False)
        #ohlc['downtrend'] = np.where(ohlc['ema_fast'] < ohlc['ema_slow'], True, False)
        return ohlc

    def open_long(self, ohlc):
        long_signal = ohlc['uptrend'].iloc[-1] 
        long_signal = long_signal and self.long_on

        return long_signal

    
    def close_position(self, ohlc, position):
        if position.status != 'open':
            raise Exception('Unable to close a closed position.')
        price = ohlc['close'].iloc[-1]
        take_profit = 1.005
        
        if (position.side == 'long') and (ohlc['close'].iloc[-1] > take_profit * position.entry_price):
            return True
        # if (position.side == 'short') and (not ohlc['downtrend'].iloc[-1]):
            # return True
        return False
