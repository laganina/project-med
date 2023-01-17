import tensorflow
import keras
import pandas as pd
import numpy as np

# ***********************************************************
# *********** RANO NEUROLOSKO POBOLJSANJE *******************
# ***********************************************************

# data = pd.read_excel(
#     r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project\podaci.xlsx')
data = pd.read_excel(r'C:\Users\Olivera\Desktop\FTN rad\podaci.xlsx')

# ****************************************
# *********** df_basic *******************
# ****************************************

# OSNOVNA OBELEZJA: 
#   - STAROST, 
#   - NIHSS na prijemu, 
#   - ASPECTS 

# data frame sa osnovnim obelezjima
basic = data[['STAROST', 'NIHSS na prijemu', 'ASPECTS', 'NIHSS 24h']]

df_basic = pd.DataFrame(basic)

df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('7 do 8', 8)         
df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('8 do 9', 9)
df_basic['ASPECTS'] = df_basic['ASPECTS'].replace('9 do 10', 10)
print('df basic:')
print(df_basic)
# ISPOD ISPISANE TABLICE CE PRIKAZATI NJENE DIMENZIJE 


# ***************************************
# *********** dodatni *******************
# ***************************************

# df sa dodatnim obelezjima
dodatni = data[['CT hiperdenzni znak','TT','Glikemija','MAP','OTT (onset to treatment time)','DNT (door to neadle time)',
                'TIP CVI','TOAST', 'ASA', 'Clopidogrel', 'OAKT', 'Statini', 'AntiHTA',
                'HTA','DM','Pušenje','HLP','Tip HLP','AA','CMP','Alkohol']]

df_dodatni = pd.DataFrame(dodatni)
print('df_dodatni:')
print(df_dodatni)

# ***************************************
# *********** lekovi *******************
# ***************************************


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


# vrednosti od 0 do 5 za da (koji lek) ne
df_lek['ASA'] = df_lek['ASA'].map({'Da': 1, 'Ne': 0})                       
df_lek['ASA'] = df_lek['ASA'].map({'da': 1, 'ne': 0})
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].map({'Da': 2, 'Ne': 0})
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].map({'da': 2, 'ne': 0})
df_lek['OAKT'] = df_lek['OAKT'].map({'Da': 3, 'Ne': 0})
df_lek['OAKT'] = df_lek['OAKT'].map({'da': 3, 'ne': 0})
df_lek['Statini'] = df_lek['Statini'].map({'Da': 4, 'Ne': 0})
df_lek['Statini'] = df_lek['Statini'].map({'da': 4, 'ne': 0})
df_lek['AntiHTA'] = df_lek['AntiHTA'].map({'Da': 5, 'Ne': 0})
df_lek['AntiHTA'] = df_lek['AntiHTA'].map({'da': 5, 'ne': 0})


# OVO RADI:

# df_lek['AntiHTA'].replace(('Da', 'Ne'), (1, 0), inplace=True)
# df_lek['AntiHTA'].replace(('da', 'ne'), (1, 0), inplace=True)




# df_lek = binary_answers('ASA', )  # KADA SVE ISTESTIRAS, MOZES DA RASPOREDI KOD PO FUNKCIJAMA DA BI BIO PREGLEDNIJI
# SVE FUNKCIJE MOZES SACUVATI U FAJL FUNKCIJE.PY I NA POCETKU OVE SKRIPTE SAMO IMPORTUJES FUNKCIJE KOJE TI TREBAJU 

print('df_lek, zamena brojevima:')
print(df_lek)

# empty cells filled with 0                                                           # KAKO CEMO RAZLIKOVATI NE ZA BILO KOJI LEK OD EMPTY CELL-A?
df_lek = df_lek.fillna(0)                                                             # TREBA IH IZBACITI NA KRAJU 
print('df_lek, zamena praznih polja nulama:')
print(df_lek)


# ***************************************
# *********** CAST TO INT  **************
# ***************************************

df_lek['ASA'] = df_lek['ASA'].astype(int)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].astype(int)
df_lek['OAKT'] = df_lek['OAKT'].astype(int)
df_lek['Statini'] = df_lek['Statini'].astype(int)
df_lek['AntiHTA'] = df_lek['AntiHTA'].astype(int)
print('df lekovi, castovanje u int:')
print(df_lek)


# svi lekovi u jednoj koloni                                          
df_lek['Lekovi'] = df_lek[df_lek.columns[0:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)

# data frame jedne kolone sa nazivom lekovi
df_l = df_lek['Lekovi']
print('df lekovi konacno:')
print(df_l)


# napravljen data frame za lekove
df_lekovi = pd.DataFrame(df_l)


# ***************************************
# *********** komorbiditeti *************
# ***************************************


# komorbiditeti izdvojeni u zaseban df
df_kom = df_dodatni[['HTA','DM','Pušenje','HLP','AA','CMP','Alkohol']].copy()

# tip hlp izdvojen u zaseban df
df_tipHLP = df_dodatni[['Tip HLP']].copy()

# svi kom izbaceni iz df_dodatni
df_dodatni = df_dodatni.drop(labels='HTA', axis=1)
df_dodatni = df_dodatni.drop(labels='DM', axis=1)
df_dodatni = df_dodatni.drop(labels='Pušenje', axis=1)
df_dodatni = df_dodatni.drop(labels='HLP', axis=1)
df_dodatni = df_dodatni.drop(labels='Tip HLP', axis=1)
df_dodatni = df_dodatni.drop(labels='AA', axis=1)
df_dodatni = df_dodatni.drop(labels='CMP', axis=1)
df_dodatni = df_dodatni.drop(labels='Alkohol', axis=1)

# hipo hdlp holesterolemija, II, nr je ne?

# vrednosti od 0 do 2 za tip HLP
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremećaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremećja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremćaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('bez poremećaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Ne', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IIa', 1)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IIb', 2)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IV', 3)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('HLP IV', 3)


# vrednosti od 0 do 7 za da (komorbiditete) ne, bez tip HLP
df_kom['HTA'] = df_kom['HTA'].map({'Da': 1, 'Ne': 0, 1:1,0:0})
df_kom['HTA'] = df_kom['HTA'].map({'da': 1, 'ne': 0, 1:1,0:0})
df_kom['DM'] = df_kom['DM'].map({'Da': 2, 'Ne': 0, 2:2,0:0})
df_kom['DM'] = df_kom['DM'].map({'da': 2, 'ne': 0, 2:2,0:0})
df_kom['Pušenje'] = df_kom['Pušenje'].map({'Da': 3, 'Ne': 0, 3:3,0:0})
df_kom['Pušenje'] = df_kom['Pušenje'].map({'da': 3, 'ne': 0, 3:3,0:0})
df_kom['HLP'] = df_kom['HLP'].map({'Da': 4, 'Ne': 0, 4:4,0:0})
df_kom['HLP'] = df_kom['HLP'].map({'da': 4, 'ne': 0, 4:4,0:0})
df_kom['AA'] = df_kom['AA'].map({'Da': 5, 'Ne': 0, 5:5,0:0})
df_kom['AA'] = df_kom['AA'].map({'da': 5, 'ne': 0, 5:5,0:0})
df_kom['CMP'] = df_kom['CMP'].map({'Da': 6, 'Ne': 0, 6:6,0:0})
df_kom['CMP'] = df_kom['CMP'].map({'da': 6, 'ne': 0, 6:6,0:0})
df_kom['Alkohol'] = df_kom['Alkohol'].map({'Da': 7, 'Ne': 0, 7:7,0:0})
df_kom['Alkohol'] = df_kom['Alkohol'].map({'da': 7, 'ne': 0, 7:7,0:0})
# empty cells filled with 0
df_kom = df_kom.fillna(0)

df_kom['HTA'] = df_kom['HTA'].astype(int)
df_kom['DM'] = df_kom['DM'].astype(int)
df_kom['Pušenje'] = df_kom['Pušenje'].astype(int)
df_kom['HLP'] = df_kom['HLP'].astype(int)
df_kom['AA'] = df_kom['AA'].astype(int)
df_kom['CMP'] = df_kom['CMP'].astype(int)
df_kom['Alkohol'] = df_kom['Alkohol'].astype(int)

# svi lekovi u jednoj koloni sa nazivom komorbiditeti
df_kom['Komorbiditeti'] = df_kom[df_kom.columns[0:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)

# data frame jedne kolone sa nazivom komorbiditeti
df_k = df_kom['Komorbiditeti']

# napravljen data frame za komorbiditete
df_komorbiditeti = pd.DataFrame(df_k)

# da ne bilo koja pretvoreno u 1 0 2
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Bilo koja', 2)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('bilo koja', 2)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Da', 1)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('da', 1)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Ne', 0)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('ne', 0)

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


# spojeni df sa lekovima
joined = df_basic.join(df_dodatni, lsuffix='_caller', rsuffix='_other')
# spojeni lekovi sa ostatkom joined
joined = pd.concat([joined, df_dodatni], axis=1)

# spojen df sa komorbiditetima
joined = df_basic.join(df_komorbiditeti, lsuffix='_caller', rsuffix='_other')
# spojeni komorbiditeti sa ostatkom joined
joined = pd.concat([joined, df_dodatni], axis=1)
# spojen tip HLP sa ostatkom joined
joined = pd.concat([joined, df_tipHLP], axis=1)




# labela za ovu klasifikaciju je pad nihss score-a za 40% nakon 24h u odnosu na inicijalnu vrednost
# napraviti varijablu nihss na prijemu, pa nihss posle 24h, labela = (nihss24h - nihssprijem)/nihssprijem,
#  ako je ta razlika kroz pocetna vrednost nihss veci ili jednak od 0.4 onda doslo je do
# poboljsanja, ako je manji od 0.4 onda je doslo do pogorsanja


# izdvojeni nihss parametri
nihssprijem = joined['NIHSS na prijemu']
nihss24 = joined['NIHSS 24h']

# napravljena lista koja ce predstavljati labelu
label = []
for i in range(len(nihssprijem)):
    label.append((nihss24[i] - nihssprijem[i]) / nihssprijem[i])

# poboljsanje = 0, pogorsanje = 1
y = []
for i in range(len(label)):
    if label[i] <= 0.4:
        y.append(0)
    else:
        y.append(1)
    i += 1

# dropping nihss 24h
df = joined.drop(labels='NIHSS 24h', axis=1)

# list to data frame
df1 = pd.DataFrame(y, columns=['STANJE'])

# stvaranje joined df sa  y iliti labelom
# x je df, y je y
merged = pd.concat([df, df1], axis=1)

# merged df sa izbacenim vrstama koje sadrze NaN value
merged = merged.dropna()

# skracivanje y tako da ima isti broj varijabli kao i ostale kolone
y = merged['STANJE']

merged = merged.astype({'STAROST':'int'})
merged = merged.astype({'NIHSS na prijemu':'int'})
merged = merged.astype({'TOAST':'int'})
merged = merged.astype({'ASPECTS':'int'})


merged = merged.T.drop_duplicates().T

merged = merged.drop(labels='Tip HLP',axis=1)
merged = merged.drop(labels='STANJE',axis=1)