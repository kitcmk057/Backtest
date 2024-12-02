import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from process.load_data.form_stock_list import get_stock_list
import hkfdb


client = hkfdb.Database('67399089')

__all__ = ['get_sec_profile']


def get_sec_profile(code_list, market, sectype, initial_capital):

    sec_profile = {}
    lot_size_dict = {}

    if market == 'HK':
        if sectype == 'STK':
            info = client.get_basic_hk_stock_info()
            for code in code_list:
                lot_size = int(info[info['code'] == code]['lot_size'])
                lot_size_dict[code] = lot_size
        else:
            for code in code_list:
                lot_size_dict[code] = 1

    elif market == 'US':
        for code in code_list:
            lot_size_dict[code] = 1

    sec_profile['market'] = market
    sec_profile['sectype'] = sectype
    sec_profile['initial_capital'] = initial_capital
    sec_profile['lot_size_dict'] = lot_size_dict

    if sectype == 'STK':
        if market == 'HK':
            sec_profile['commission_rate'] = 0.03 * 0.01
            sec_profile['min_commission'] = 3
            sec_profile['platform_fee'] = 15
        if market == 'US':
            sec_profile['commission_each_stock']   = 0.0049
            sec_profile['min_commission']          = 0.99
            sec_profile['platform_fee_each_stock'] = 0.005
            sec_profile['min_platform_fee']        = 1

    return sec_profile

