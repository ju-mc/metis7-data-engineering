#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 00:38:26 2021

@author: juju
"""

import twint

def get_tweets(keyword):
    c = twint.Config()
    c.Search = keyword
    c.Lang = "en"
    c.Limit = 50000
    #c.Output = "./test.json"
    c.Output='./{}_data.json'.format(keyword)
    c.Store_json = True
    
    twint.run.Search(c)
    return c.Output

'''keyword_list=['NFT','CryptoPunks','Bored Ape Club','Art Blocks Curated','Bored Ape Kennel Club',
              'Cool Cats NFT','Zed Run','Meebits','FameLady Squad','Curio.Cards','SupDucks']'''

'''keyword_list=['Rarible','Chubbies NFT','Avastars','Ghxsts','CyberKongz_old']'''
    
for keyword in keyword_list:
    get_tweets(keyword)
    



