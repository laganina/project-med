# import tensorflow
# import keras
import pandas as pd
import numpy as np

# ***********************************************************
# *********** RANO NEUROLOSKO POBOLJSANJE *******************
# ***********************************************************

# data = pd.read_excel(r'C:\Users\Olivera\Documents\PythonScripts\SlogOporavakProjekat2022-23\project-med\project\podaci.xlsx')    # ZAMENI PUTANJU 
data = pd.read_excel(r'C:\Users\Olivera\Desktop\FTN rad\Izrada\PROJEKAT\Olivera\podaci.xlsx')    # ZAMENI PUTANJU

# ****************************************
# *********** df_basic *******************
# ****************************************

# data frame sa osnovnim obelezjima
basic = data[['STAROST', 'NIHSS na prijemu', 'ASPECTS', 'NIHSS 24h']]
# - Starost - mladji pacijenti, veca sansa za akutno poboljsanje
# - Nihss na prijemu je korelisan sa tezinom mozdanog udara - blazi mozdani udar, niza vrednost ovog parametra
# - ASPECT score - radioloski parametar 
# - Nihss 24 h koristi se za izvodjenje zakljucka o ranom neuroloskom poboljsanju 



df_basic = pd.DataFrame(basic)

df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('7 do 8', 8)
df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('8 do 9', 9)
df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('9 do 10', 10)
print('df basic:')
print(df_basic)
# ISPOD ISPISANE TABLICE CE PRIKAZATI NJENE DIMENZIJE
print('*****************************************************************************')   
print('*****************************************************************************') 

# ***************************************
# *********** dodatni *******************
# ***************************************

# df sa dodatnim obelezjima
# ovde su i LEKOVI od ASA - ANTI HTA
# ovde su i KOMORBIDITETI od HTA - Alkohol 

# MOZEMO DA IH GRUPISEMO OVAKO:   # LEKOVI I KOMORBIDITETI NA STRANU 
# LEKOVI: 'ASA', 'Clopidogrel', 'OAKT', 'Statini', 'AntiHTA'
# KOMORBIDITETI: 'HTA','DM','Pušenje','HLP','Tip HLP','AA','CMP','Alkohol' ******** bez tip HLP (HLP je )

# 'CT hiperdenzni znak'
# TT - telesna masa
# OTT i DNT su vremena
# TIP CVI i TOAST - tip udara i uzrok 



dodatni = data[['CT hiperdenzni znak','TT','Glikemija','MAP','OTT (onset to treatment time)','DNT (door to neadle time)',
                'TIP CVI','TOAST', 'ASA', 'Clopidogrel', 'OAKT', 'Statini', 'AntiHTA',
                'HTA','DM','Pušenje','HLP','Tip HLP','AA','CMP','Alkohol']]

df_dodatni = pd.DataFrame(dodatni)

print('df_dodatni:')
print(df_dodatni)
print('*****************************************************************************')   
print('*****************************************************************************') 

# **************************************
# *********** LEKOVI ******************* # **************************************** TREBALO BI MENJATI SA 1, SVUDA
# **************************************
# ASA - ANTI HTA

# df za lekove izdvojen iz df_dodatni
df_lek = df_dodatni[['ASA', 'Clopidogrel','OAKT', 'Statini', 'AntiHTA']].copy()

print('df_lek, tek ucitano:')
print(df_lek)

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

# vrednosti od 0 do 5 za da (koji lek) ne
df_lek['ASA'] = df_lek['ASA'].replace('Da', 1)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('Da', 1)     # trebalo bi da stoji 1, svuda 
df_lek['OAKT'] = df_lek['OAKT'].replace('Da', 1)
df_lek['Statini'] = df_lek['Statini'].replace('Da', 1)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('Da', 1)

df_lek['ASA'] = df_lek['ASA'].replace('da', 1)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('da', 1)    
df_lek['OAKT'] = df_lek['OAKT'].replace('da', 1)
df_lek['Statini'] = df_lek['Statini'].replace('da', 1)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('da', 1)

df_lek['ASA'] = df_lek['ASA'].replace('Ne', 0)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('Ne', 0)
df_lek['OAKT'] = df_lek['OAKT'].replace('Ne', 0)
df_lek['Statini'] = df_lek['Statini'].replace('Ne', 0)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('Ne', 0)

df_lek['ASA'] = df_lek['ASA'].replace('ne', 0)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('ne', 0)
df_lek['OAKT'] = df_lek['OAKT'].replace('ne', 0)
df_lek['Statini'] = df_lek['Statini'].replace('ne', 0)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('ne', 0)



print('df_lek, zamena brojevima:')
print(df_lek)
# *** Sada nam ostaje 5 nezavisnih kolona gde je u svakoj jedno obelezje. Ako cemo ih tako ostavljati onda, to mogu sve da budu binarne varijable.
# DRUGO RESENJE JE DA IH UKLOPIMO SVE U JEDNO OBELEZJE CIJA CE VREDNOST DA BUDE DECIMALNI EKVIVALENT BINARNOG BROJA SASTAVLJENOG OD OVIH 5 OBELEZJA 
print('*****************************************************************************')   
print('*****************************************************************************') 

# ***************************************
# *********** KOMORBIDITETI ************* 
# ***************************************
# HTA - Alkohol 

# komorbiditeti izdvojeni u zaseban df
df_kom = df_dodatni[['HTA','DM','Pušenje','HLP','AA','CMP','Alkohol']].copy()

# tip hlp izdvojen u zaseban df
df_tipHLP = df_dodatni[['Tip HLP']].copy()

# svi kom izbaceni iz df_dodatni
df_dodatni = df_dodatni.drop(labels='HTA', axis=1)             # DA/NE -> 1/0 
df_dodatni = df_dodatni.drop(labels='DM', axis=1)              # DA/NE
df_dodatni = df_dodatni.drop(labels='Pušenje', axis=1)         # DA/NE 
df_dodatni = df_dodatni.drop(labels='HLP', axis=1)             # DA/NE
df_dodatni = df_dodatni.drop(labels='Tip HLP', axis=1)         # bez poremecaja, 2a, 2b, 4 -> 0,1,2,3 NIJE BINARNA VARIJABLA, MOZEMO JE IZDVOJITI KAO POSEBNO OBELEZJE 
df_dodatni = df_dodatni.drop(labels='AA', axis=1)              # DA/NE
df_dodatni = df_dodatni.drop(labels='CMP', axis=1)             # DA/NE
df_dodatni = df_dodatni.drop(labels='Alkohol', axis=1)         # DA/NE

# ******************************************
# *********** KOMORBIDITETI - TIP HLP ******
# ******************************************

# hipo hdlp holesterolemija, II, nr je ne?
# SVE MOGUCE VREDNOSTI DESKRIPTORA DATE SU U TABELI U EXCELU U SHEETU 'NE DIRATI'

# vrednosti od 0 do 3 za tip HLP - ODLICNO ODRADJENO 
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremećaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremećja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremćaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('bez poremećaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Ne', 0)

df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IIa', 1)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IIb', 2)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IV', 3)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('HLP IV', 3)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('nr', np.nan)   # OKEJ, STAVLJAMO NAN TAMO GDE NEMAMO PODATKE, POSLE IZBACUJEMO SVE NANOVE U JEDNOM KORAKU, NA KRAJU 
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('n/a', np.nan)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Hipo HDL holesterolemija', np.nan)


print('df_tipHLP, zamena brojevima:')
print(df_tipHLP)                              # OKEJ 
print('*****************************************************************************')   
print('*****************************************************************************') 

# ***************************************************
# *********** KOMORBIDITETI - ZAMENA BROJEVIMA ****** 
# ***************************************************



# vrednosti od 0 do 7 za da (komorbiditete) ne, bez tip HLP - KOMENTARI SU ISTI KAO I ZA LEKOVE 
df_kom['HTA'] = df_kom['HTA'].replace('Da', 1)
df_kom['HTA'] = df_kom['HTA'].replace('da', 1)

df_kom['DM'] = df_kom['DM'].replace('Da', 1)        
df_kom['DM'] = df_kom['DM'].replace('da', 1)

df_kom['Pušenje'] = df_kom['Pušenje'].replace('Da', 1)
df_kom['Pušenje'] = df_kom['Pušenje'].replace('da', 1)

df_kom['HLP'] = df_kom['HLP'].replace('Da', 1)
df_kom['HLP'] = df_kom['HLP'].replace('da', 1)

df_kom['AA'] = df_kom['AA'].replace('Da', 1)
df_kom['AA'] = df_kom['AA'].replace('da', 1)

df_kom['CMP'] = df_kom['CMP'].replace('Da', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da ', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertrophica comp)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP ischaemica9', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP ischaemica)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvulars)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvularis chr. Com EF 45%)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvularis chr. Com)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa, EF 23%)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertensiva)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP isch)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertrophica comp)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa EF 35%)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('da', 1)
df_kom['CMP'] = df_kom['CMP'].replace('da (CMP hypertensiva hypertrophica comp)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('da (CMP ischaemica)', 1)
df_kom['CMP'] = df_kom['CMP'].replace('da, CMP valvularis', 1)

df_kom['Alkohol'] = df_kom['Alkohol'].replace('Da', 1)
df_kom['Alkohol'] = df_kom['Alkohol'].replace('Da', 1)

# NE  


df_kom['HTA'] = df_kom['HTA'].replace('Ne', 0)
df_kom['HTA'] = df_kom['HTA'].replace('ne', 0)

df_kom['DM'] = df_kom['DM'].replace('Ne', 0)
df_kom['DM'] = df_kom['DM'].replace('ne', 0)

df_kom['Pušenje'] = df_kom['Pušenje'].replace('Ne', 0)
df_kom['Pušenje'] = df_kom['Pušenje'].replace('ne', 0)

df_kom['HLP'] = df_kom['HLP'].replace('Ne', 0)
df_kom['HLP'] = df_kom['HLP'].replace('ne', 0)
df_kom['HLP'] = df_kom['HLP'].replace('nr', np.nan)    # NAN 

df_kom['AA'] = df_kom['AA'].replace('Ne', 0)
df_kom['AA'] = df_kom['AA'].replace('ne', 0)

df_kom['CMP'] = df_kom['CMP'].replace('Ne', 0)
df_kom['CMP'] = df_kom['CMP'].replace('ne', 0)

df_kom['Alkohol'] = df_kom['Alkohol'].replace('Ne', 0)
df_kom['Alkohol'] = df_kom['Alkohol'].replace('ne', 0)

print('df_kom, zamena brojevima:')
print(df_kom)
print('*****************************************************************************')   
print('*****************************************************************************') 

# **************************************
# *********** CT hiperdenzni znak ******
# **************************************

# da ne bilo koja pretvoreno u 1 0 2
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Bilo koja', 2)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('bilo koja', 2)

df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Da', 1)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('da', 1)

df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Ne', 0)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('ne', 0)

print('df_dodatni, CT, zamena brojevima:')
print(df_dodatni[['CT hiperdenzni znak']])
print('*****************************************************************************')   
print('*****************************************************************************') 

# ************************************
# *********** TIP CVI ****************
# ************************************

# tip cvi - pretvoreni u 0 1 2 3
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI', 0)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI    ', 0)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI  ', 0)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI   ', 0)

df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('PACI', 1)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('PACI  ', 1)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('PACI ', 1)

df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LAC', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI ', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI  ', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI?', 2)

df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('POCI', 3)

print('df_dodatni, tip cvi, zamena brojevima:')
print(df_dodatni[['TIP CVI']])
print('*****************************************************************************')   
print('*****************************************************************************') 


# **********************************
# *********** TOAST ****************
# **********************************

# tip tost - pretvoreni u 0 1 2 3 4 5
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('LAA', 0)

df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('CE', 1)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('CE?', 1)

df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('SVD', 2)

df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Drugi', 3)

df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Neutvrđeno', 4)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Neutrvđeno', 4)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Neutvrđen', 4)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Neutrvđen', 4)

df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Stroke mimic', 5)

print('df_dodatni, TOAST, zamena brojevima:')
print(df_dodatni[['TOAST']])
print('*****************************************************************************')   
print('*****************************************************************************') 



# ***************************************************
# *********** SPAJANJE DATAFRAME-OVA ****************
# ***************************************************

# napravljen data frame za komorbiditete
df_kom = pd.DataFrame(df_kom)

# spojeni df sa lekovima - BASIC + DODATNI
joined = df_basic.join(df_dodatni, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni: ')
print(joined)

# spojeni lekovi sa ostatkom joined - BASIC + DODATNI + LEKOVI 
# joined = joined.join([df_lek], lsuffix='_caller', rsuffix='_other')
joined = joined.join(df_lek, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi: ')
print(joined)

# spojen tip HLP sa ostatkom joined - BASIC + DODATNI + LEKOVI + TIP HLP 
# joined = joined.join([df_tipHLP], lsuffix='_caller', rsuffix='_other')
joined = joined.join(df_tipHLP, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi + tip HLP: ')

#joined df_komorbiditeti sa df_dodatni - BASIC + DODATNI + LEKOVI + TIP HLP + KOMORB. 
# joined = joined.join([df_kom], lsuffix='_caller', rsuffix='_other')
joined = joined.join(df_kom, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi + tip HLP + komorbiditeti: ')

print('sve spojeno: ')
print(joined)
print(f'DIMENZIJE CELOG DF: {joined.size} ')
print('*****************************************************************************')   
print('*****************************************************************************') 



# DROPOVACEMO TIP HLP
joined = joined.drop(labels='Tip HLP',axis=1)    

joined = joined.fillna(np.nan)   # ako su prazni, dodeljuje nan
joined = joined.dropna()         # izbacuje sve nanove 
joined = joined.reset_index(drop=True)   # resetuje indekse, potrebno zbog izbacenih nanova  
print('index kolona:')
print(joined.index)
joined.index.name = 'index'
print(f'index name: {joined.index.name}')
print(joined.info())
print(joined)


print(f'DIMENZIJE DF POSLE IZBACIVANJA NANOVA, drugi put: {joined.shape} ')
# **********************************************************************
print('*****************************************************************************')   
print('*****************************************************************************') 

# lekovi, komorbiditeti, starost idu u int 
joined['ASA'] = joined['ASA'].astype(int)
joined['Clopidogrel'] = joined['Clopidogrel'].astype(int)
joined['OAKT'] = joined['OAKT'].astype(int)
joined['Statini'] = joined['Statini'].astype(int)
joined['AntiHTA'] = joined['AntiHTA'].astype(int)

# komorbiditeti: 
joined = joined.astype({'HTA' :int})
joined = joined.astype({'DM' :int})
joined = joined.astype({'Pušenje' :int})
joined = joined.astype({'HLP' :int})
joined = joined.astype({'AA' :int})
joined = joined.astype({'CMP' :int})
joined = joined.astype({'Alkohol' :int})


# starost, toast, tipcvi, tip hlp, aspect score, CT hiperdenzni znak
# joined = joined.astype({'STAROST' : float})
# joined = joined.astype({'Tip HLP' :int})
joined = joined.astype({'ASPECTS' :int})
joined = joined.astype({'TIP CVI' :int})
joined = joined.astype({'CT hiperdenzni znak' :int})
joined = joined.astype({'TOAST' :int})
joined = joined.astype({'OTT (onset to treatment time)' :float})    # zbog standard scaler-a neka ostanu float 


print(joined.info())
# ********************************************************************************************************
# ********************************************************************************************************
joined_basic = joined[['STAROST', 'NIHSS na prijemu', 'ASPECTS']]
print('Ista memorija:')
print(joined_basic is joined)

starost = joined_basic['STAROST']
print(f'STAROST: {starost}')


# sada nadji labelu 

# **************************************
# *********** LABELA *******************
# **************************************
# AKO PADNE VISE OD 40 POSTO ONDA JE DOSLO DO POBOLJSANJA 
#  

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
# for i in nihssprijem:   # ovo nije dobro 
for i in range(len(nihssprijem)):
    label.append((nihssprijem[i] - nihss24[i]) / nihssprijem[i])
print('label:')
print(label)
print(len(label))


# poboljsanje = 0, pogorsanje = 1   # POBOLJSANJE JE 1, POGORSANJE JE 0 
y = []

for i in range(len(label)):   
    if label[i] < 0.4:
        y.append(0)
    else:
        y.append(1)
    # i += 1

print('y:')
# print(label)
print(len(y))
print(type(y))
print(y)

# dropping nihss 24h
obelezja = joined.drop(labels='NIHSS 24h', axis=1)
print(f'Dimenzije obelezja, {obelezja.shape}')             # 388x19
print('*****************************************************************************')   
print('*****************************************************************************') 


# list to data frame
labela = pd.DataFrame(y, columns=['STANJE'])   # DF1 SADRZI STANJE, TJ LABELU  
print(f'Dimenzije labela, {labela.shape}')             # 388x1 
print(labela)
print(labela.info())
broj_jedinica = sum(labela['STANJE'] == 1)
broj_nula = sum(labela['STANJE'] == 0)
print(f'broj_jedinica: {broj_jedinica}')
print(f'broj_nula: {broj_nula}')

# broj_jedinica: 163
# broj_nula: 230

# print('*****************************************************************************')   
# print('*****************************************************************************') 

# float su:
# STAROST, NIHSS, NIHSS24, TT MASA, GLIKEMIJA, MAP, OTT, DNT. 
# nisu diskretne varijable 