# -*- coding: utf-8 -*-
import txt_to_df
import parse_and_sen


path=r'C:\Users\Z\Desktop\NI\engdata\Narratives_valid324.txt'

nar_df = txt_to_df(path)
nar_df = parse(path)
nar_df = sentiment(nar_df)