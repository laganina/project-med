import tensorflow
import keras
import pandas as pd
import numpy as np
import toast
import lekovi
import ct
import tip_cvi
import komorbiditeti_prvi
import tip_hlp
from lekovi import df_lek
from komorbiditeti_prvi import df_kom
from tip_hlp import df_tipHLP


#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
# RANO NEULOSKO OBOLJENJE
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#



# zameni putanju!!!!
data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')

#----------------------------------------------------------------------------------------------------------------------#
# DF_BASIC
# data frame sa osnovnim obelezjima

#**********************************************************************************************************************#
# - Starost - mladji pacijenti, veca sansa za akutno poboljsanje
# - Nihss na prijemu je korelisan sa tezinom mozdanog udara - blazi mozdani udar, niza vrednost ovog parametra
# - ASPECT score - radioloski parametar
# - Nihss 24 h koristi se za izvodjenje zakljucka o ranom neuroloskom poboljsanju
#**********************************************************************************************************************#


basic = data[['STAROST', 'NIHSS na prijemu', 'ASPECTS', 'NIHSS 24h']]
df_basic = pd.DataFrame(basic)

df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('7 do 8', 8)
df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('8 do 9', 9)
df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('9 do 10', 10)
print('df basic:')
print(df_basic)

# dimenzije df_basic
print('*****************************************************************************')
print('*****************************************************************************')

#----------------------------------------------------------------------------------------------------------------------
# DODATNI
# CT hiperdenzni znak, TT - telesna masa, OTT i DNT - vremena
# tip CVI i TOAST - tip i uzrok mozdanog udara

# (dodatna obelezja sa lekovima ASA->antiHTA i komorbiditetima HTA->Alkohol)
# tip HLP ce biti izdvojeni zbog razlicitih tipova

#----------------------------------------------------------------------------------------------------------------------#
dodatni = data[['CT hiperdenzni znak','TT','Glikemija','MAP','OTT (onset to treatment time)','DNT (door to neadle time)',
                'TIP CVI','TOAST', 'ASA', 'Clopidogrel', 'OAKT', 'Statini', 'AntiHTA',
                'HTA','DM','Pušenje','HLP','Tip HLP','AA','CMP','Alkohol']]
df_dodatni = pd.DataFrame(dodatni)

print('df_dodatni:')
print(df_dodatni)
print('*****************************************************************************')
print('*****************************************************************************')

#----------------------------------------------------------------------------------------------------------------------#
# svi lekovi izbaceni iz df_dodatni
df_dodatni = df_dodatni.drop(labels='ASA', axis=1)
df_dodatni = df_dodatni.drop(labels='Clopidogrel', axis=1)
df_dodatni = df_dodatni.drop(labels='OAKT', axis=1)
df_dodatni = df_dodatni.drop(labels='Statini', axis=1)
df_dodatni = df_dodatni.drop(labels='AntiHTA', axis=1)

print('df_dodatni posle izbacivanja lekova:')
print(df_dodatni)
print('*****************************************************************************')
print('*****************************************************************************')

#----------------------------------------------------------------------------------------------------------------------#
# svi kom izbaceni iz df_dodatni
df_dodatni = df_dodatni.drop(labels='HTA', axis=1)             # DA/NE -> 1/0
df_dodatni = df_dodatni.drop(labels='DM', axis=1)              # DA/NE
df_dodatni = df_dodatni.drop(labels='Pušenje', axis=1)         # DA/NE
df_dodatni = df_dodatni.drop(labels='HLP', axis=1)             # DA/NE
df_dodatni = df_dodatni.drop(labels='Tip HLP', axis=1)         # bez poremecaja, 2a, 2b, 4
df_dodatni = df_dodatni.drop(labels='AA', axis=1)              # DA/NE
df_dodatni = df_dodatni.drop(labels='CMP', axis=1)             # DA/NE
df_dodatni = df_dodatni.drop(labels='Alkohol', axis=1)         # DA/NE


#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
# SPAJANJE DATAFRAME-OVA
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#


# BASIC + DODATNI
joined = df_basic.join(df_dodatni, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni: ')
print(joined)

# BASIC + DODATNI + LEKOVI
joined = joined.join(df_lek, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi: ')
print(joined)

# BASIC + DODATNI + LEKOVI + TIP HLP + KOMORB.
joined = joined.join(df_kom, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi + tip HLP + komorbiditeti: ')

print('sve spojeno: ')
print(joined)
print(f'DIMENZIJE CELOG DF: {joined.size} ')
print('*****************************************************************************')
print('*****************************************************************************')

# BASIC + DODATNI + LEKOVI + TIP HLP
joined = joined.join(df_tipHLP, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi + tip HLP: ')


#----------------------------------------------------------------------------------------------------------------------#
# SREDJIVANJE JOINED (BASIC + DODATNI + LEKOVI + TIP HLP + KOMORBIDITETI)

joined = joined.drop(labels='Tip HLP',axis=1)
# ako su prazni, dodeljuje NaN
joined = joined.fillna(np.nan)
# izbacuje NaN
joined = joined.dropna()
# resetuje indekse, potrebno zbog izbacenih nanova
joined = joined.reset_index(drop=True)
print('index kolona:')
print(joined.index)
joined.index.name = 'index'
print(f'index name: {joined.index.name}')
print(joined.info())
print(joined)
print(f'DIMENZIJE DF POSLE IZBACIVANJA NANOVA, drugi put: {joined.shape} ')

print('*****************************************************************************')
print('*****************************************************************************')

#----------------------------------------------------------------------------------------------------------------------#
# lekovi u int
joined['ASA'] = joined['ASA'].astype(int)
joined['Clopidogrel'] = joined['Clopidogrel'].astype(int)
joined['OAKT'] = joined['OAKT'].astype(int)
joined['Statini'] = joined['Statini'].astype(int)
joined['AntiHTA'] = joined['AntiHTA'].astype(int)
# ct u int
df_dodatni = df_dodatni.astype({'CT hiperdenzni znak' :int})
# aspect score, ott u float
joined = joined.astype({'ASPECTS' :int})
# tip cvi u int
joined = joined.astype({'TIP CVI' :int})
# ott u float
joined = joined.astype({'OTT (onset to treatment time)' :float})
# komorbiditeti u int:
df_kom = df_kom.astype({'HTA' :int})
df_kom = df_kom.astype({'DM' :int})
df_kom = df_kom.astype({'Pušenje' :int})
df_kom = df_kom.astype({'HLP' :int})
df_kom = df_kom.astype({'AA' :int})
df_kom = df_kom.astype({'CMP' :int})
df_kom = df_kom.astype({'Alkohol' :int})

print(joined.info())

#----------------------------------------------------------------------------------------------------------------------#
# info za starost, nihss na prijemu aspects
joined_basic = joined[['STAROST', 'NIHSS na prijemu', 'ASPECTS']]
print('Ista memorija:')
print(joined_basic is joined)

# info za starost
starost = joined_basic['STAROST']
print(f'STAROST: {starost}')

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
# LABELA
# <40 poboljsanje, >=40 pogorsanje
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

# izdvojeni nihss parametri
nihssprijem = joined['NIHSS na prijemu']
print(f'Dimenzije nihssprijem, {nihssprijem.shape}')
print('nihssprijem:')
print(nihssprijem)
print('*****************************************************************************')
print('*****************************************************************************')

nihss24 = joined['NIHSS 24h']
print(f'Dimenzije nihss24, {nihss24.shape}')
print('nihss24:')
print(nihss24)
print('*****************************************************************************')
print('*****************************************************************************')

# napravljena lista koja ce predstavljati labelu
label = []

# ?????????? da li je prepravljeno  ??????????

# for i in nihssprijem:   # ovo nije dobro
for i in range(len(nihssprijem)):
    label.append((nihssprijem[i] - nihss24[i]) / nihssprijem[i])
print('label:')
print(label)
print(len(label))


# poboljsanje = 0,
# pogorsanje = 1

y = []
for i in range(len(label)):
    if label[i] < 0.4:
        y.append(0)
    else:
        y.append(1)

print('y:')
print(len(y))
print(type(y))
print(y)

#----------------------------------------------------------------------------------------------------------------------#

# dropping nihss 24h
obelezja = joined.drop(labels='NIHSS 24h', axis=1)
print(f'Dimenzije obelezja, {obelezja.shape}')             # 388x19
print('*****************************************************************************')
print('*****************************************************************************')

#----------------------------------------------------------------------------------------------------------------------#
# list to data frame
labela = pd.DataFrame(y, columns=['STANJE'])   # df1 sadezi stanje, tj labelu
print(f'Dimenzije labela, {labela.shape}')     # 388x1
print(labela)
print(labela.info())
broj_jedinica = sum(labela['STANJE'] == 1)
broj_nula = sum(labela['STANJE'] == 0)
print(f'broj_jedinica: {broj_jedinica}')
print(f'broj_nula: {broj_nula}')


#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

# broj_jedinica: 163
# broj_nula: 230

# print('*****************************************************************************')
# print('*****************************************************************************')

# float su:
# STAROST, NIHSS, NIHSS24, TT MASA, GLIKEMIJA, MAP, OTT, DNT.
# nisu diskretne varijable

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
