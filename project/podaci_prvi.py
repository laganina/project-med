# import tensorflow
# import keras
import pandas as pd
import numpy as np

# ***********************************************************
# *********** RANO NEUROLOSKO POBOLJSANJE *******************
# ***********************************************************

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
# ovde su i lekovi od ASA - ANTI HTA
# ovde su i komorbiditeti od HTA - Alkohol 
dodatni = data[['CT hiperdenzni znak','TT','Glikemija','MAP','OTT (onset to treatment time)','DNT (door to neadle time)',
                'TIP CVI','TOAST', 'ASA', 'Clopidogrel', 'OAKT', 'Statini', 'AntiHTA',
                'HTA','DM','Pušenje','HLP','Tip HLP','AA','CMP','Alkohol']]

df_dodatni = pd.DataFrame(dodatni)

print('df_dodatni:')
print(df_dodatni)

# **************************************
# *********** LEKOVI *******************
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

# vrednosti od 0 do 5 za da (koji lek) ne
df_lek['ASA'] = df_lek['ASA'].replace('Da', 1)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('Da', 2)
df_lek['OAKT'] = df_lek['OAKT'].replace('Da', 3)
df_lek['Statini'] = df_lek['Statini'].replace('Da', 4)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('Da', 5)

df_lek['ASA'] = df_lek['ASA'].replace('da', 1)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('da', 2)
df_lek['OAKT'] = df_lek['OAKT'].replace('da', 3)
df_lek['Statini'] = df_lek['Statini'].replace('da', 4)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('da', 5)

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

# vrednosti od 0 do 2 za tip HLP - ODLICNO ODRADJENO 
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

print('df_tipHLP, zamena brojevima:')
print(df_tipHLP)                              # OKEJ 

# ***************************************************
# *********** KOMORBIDITETI - ZAMENA BROJEVIMA ******
# ***************************************************



# vrednosti od 0 do 7 za da (komorbiditete) ne, bez tip HLP - KOMENTARI SU ISTI KAO I ZA LEKOVE 
df_kom['HTA'] = df_kom['HTA'].replace('Da', 1)
df_kom['HTA'] = df_kom['HTA'].replace('da', 1)

df_kom['DM'] = df_kom['DM'].replace('Da', 2)
df_kom['DM'] = df_kom['DM'].replace('da', 2)

df_kom['Pušenje'] = df_kom['Pušenje'].replace('Da', 3)
df_kom['Pušenje'] = df_kom['Pušenje'].replace('da', 3)

df_kom['HLP'] = df_kom['HLP'].replace('Da', 4)
df_kom['HLP'] = df_kom['HLP'].replace('da', 4)

df_kom['AA'] = df_kom['AA'].replace('Da', 5)
df_kom['AA'] = df_kom['AA'].replace('da', 5)

df_kom['CMP'] = df_kom['CMP'].replace('Da', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da ', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertrophica comp)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP ischaemica9', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP ischaemica)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvulars)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvularis chr. Com EF 45%)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvularis chr. Com)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa, EF 23%)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertensiva)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP isch)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertrophica comp)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa EF 35%)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('da', 6)
df_kom['CMP'] = df_kom['CMP'].replace('da (CMP hypertensiva hypertrophica comp)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('da (CMP ischaemica)', 6)
df_kom['CMP'] = df_kom['CMP'].replace('da, CMP valvularis', 6)

df_kom['Alkohol'] = df_kom['Alkohol'].replace('Da', 7)
df_kom['Alkohol'] = df_kom['Alkohol'].replace('Da', 7)

df_kom['HTA'] = df_kom['HTA'].replace('Ne', 0)
df_kom['HTA'] = df_kom['HTA'].replace('ne', 0)

df_kom['DM'] = df_kom['DM'].replace('Ne', 0)
df_kom['DM'] = df_kom['DM'].replace('ne', 0)

df_kom['Pušenje'] = df_kom['Pušenje'].replace('Ne', 0)
df_kom['Pušenje'] = df_kom['Pušenje'].replace('ne', 0)

df_kom['HLP'] = df_kom['HLP'].replace('Ne', 0)
df_kom['HLP'] = df_kom['HLP'].replace('ne', 0)
df_kom['HLP'] = df_kom['HLP'].replace('nr', np.nan)

df_kom['AA'] = df_kom['AA'].replace('Ne', 0)
df_kom['AA'] = df_kom['AA'].replace('ne', 0)
df_kom['CMP'] = df_kom['CMP'].replace('Ne', 0)
df_kom['CMP'] = df_kom['CMP'].replace('ne', 0)

df_kom['Alkohol'] = df_kom['Alkohol'].replace('Ne', 0)
df_kom['Alkohol'] = df_kom['Alkohol'].replace('ne', 0)

print('df_kom, zamena brojevima:')
print(df_kom)

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




# ***************************************************
# *********** SPAJANJE DATAFRAME-OVA ****************
# ***************************************************

# napravljen data frame za komorbiditete
df_kom = pd.DataFrame(df_kom)

# spojeni df sa lekovima - BASIC + DODATNI
joined = df_basic.join(df_dodatni, lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni: ')
print(joined)

# spojeni lekovi sa ostatkom joined
joined = joined.join([df_lek], lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi: ')
print(joined)

# spojen tip HLP sa ostatkom joined
joined = joined.join([df_tipHLP], lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi + tip HLP: ')

#joined df_komorbiditeti sa df_dodatni
joined = joined.join([df_kom], lsuffix='_caller', rsuffix='_other')
print('df basic + dodatni + lekovi + tip HLP + komorbiditeti: ')

print('sve spojeno: ')
print(joined)
print(f'DIMENZIJE CELOG DF: {joined.shape} ')


# dropped nan values
joined = joined.dropna()

print('bez nan val sve spojeno: ')
print(joined)
print(f'DIMENZIJE DF POSLE IZBACIVANJA NANOVA: {joined.shape} ')




# *******************************************************
# *********** KOMORBIDITETI U JEDNOJ KOLONI *************
# *******************************************************

# kom izbaceni u zaseban df
df_komorbiditeti = joined[['HTA', 'DM', 'Pušenje', 'HLP', 'AA', 'CMP', 'Alkohol']].copy()

# ***************************************
# *********** CAST TO INT  **************
# ***************************************

df_komorbiditeti = df_komorbiditeti.astype({'HTA' :int})
df_komorbiditeti = df_komorbiditeti.astype({'DM' :int})
df_komorbiditeti = df_komorbiditeti.astype({'Pušenje' :int})
df_komorbiditeti = df_komorbiditeti.astype({'HLP' :int})
df_komorbiditeti = df_komorbiditeti.astype({'AA' :int})
df_komorbiditeti = df_komorbiditeti.astype({'CMP' :int})
df_komorbiditeti = df_komorbiditeti.astype({'Alkohol' :int})

# svi komorbiditeti u jednoj koloni sa nazivom komorbiditeti
df_komorbiditeti['Komorbiditeti'] = df_komorbiditeti[df_komorbiditeti.columns[0:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)


df_komorbiditeti = df_komorbiditeti.drop(labels='HTA', axis=1)
df_komorbiditeti = df_komorbiditeti.drop(labels='DM', axis=1)
df_komorbiditeti = df_komorbiditeti.drop(labels='Pušenje', axis=1)
df_komorbiditeti = df_komorbiditeti.drop(labels='HLP', axis=1)
df_komorbiditeti = df_komorbiditeti.drop(labels='AA', axis=1)
df_komorbiditeti = df_komorbiditeti.drop(labels='CMP', axis=1)
df_komorbiditeti = df_komorbiditeti.drop(labels='Alkohol', axis=1)


joined = joined.join([df_komorbiditeti], lsuffix='_caller', rsuffix='_other')
joined = joined.drop(labels='HTA', axis=1)
joined = joined.drop(labels='DM', axis=1)
joined = joined.drop(labels='Pušenje', axis=1)
joined = joined.drop(labels='HLP', axis=1)
joined = joined.drop(labels='AA', axis=1)
joined = joined.drop(labels='CMP', axis=1)
joined = joined.drop(labels='Alkohol', axis=1)

print('joined samo sa komorbiditeti: ')
print(joined)

# ******************************************************
# *********** LEKOVI U JEDNOJ KOLONI *******************
# ******************************************************

df_lekovi = joined[['ASA', 'Clopidogrel', 'OAKT', 'Statini', 'AntiHTA']].copy()

# ***************************************
# *********** CAST TO INT  **************
# ***************************************

df_lekovi['ASA'] = df_lekovi['ASA'].astype(int)
df_lekovi['Clopidogrel'] = df_lekovi['Clopidogrel'].astype(int)
df_lekovi['OAKT'] = df_lekovi['OAKT'].astype(int)
df_lekovi['Statini'] = df_lekovi['Statini'].astype(int)
df_lekovi['AntiHTA'] = df_lekovi['AntiHTA'].astype(int)

# svi lekovi u jednoj koloni
df_lekovi['Lekovi'] = df_lekovi[df_lek.columns[0:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)

df_lekovi = df_lekovi.drop(labels='ASA',axis=1)
df_lekovi = df_lekovi.drop(labels='Clopidogrel',axis=1)
df_lekovi = df_lekovi.drop(labels='OAKT',axis=1)
df_lekovi = df_lekovi.drop(labels='Statini',axis=1)
df_lekovi = df_lekovi.drop(labels='AntiHTA',axis=1)


joined = joined.join([df_lekovi], lsuffix='_caller', rsuffix='_other')

print('samo s lekovi: ')
print(joined)


# **************************************
# *********** LABELA *******************
# **************************************

# labela za ovu klasifikaciju je pad nihss score-a za 40% nakon 24h u odnosu na inicijalnu vrednost
# napraviti varijablu nihss na prijemu, pa nihss posle 24h, labela = (nihss24h - nihssprijem)/nihssprijem,
#  ako je ta razlika kroz pocetna vrednost nihss veci ili jednak od 0.4 onda doslo je do
# poboljsanja, ako je manji od 0.4 onda je doslo do pogorsanja

joined = joined.astype({'NIHSS na prijemu':'int'})
joined = joined.astype({'NIHSS 24h':'int'})

# izdvojeni nihss parametri
nihssprijem = joined['NIHSS na prijemu']
nihss24 = joined['NIHSS 24h']

# napravljena lista koja ce predstavljati labelu
label = []
for i in nihssprijem:
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

print('joined sa score: ')
print(joined)

# stvaranje joined df sa y iliti labelom
# x je df, y je y
merged = pd.concat([df, df1], axis=1)

print('merged sa y, bez nihss 24h: ')
print(merged)

# merged df sa izbacenim vrstama koje sadrze NaN value
merged = merged.dropna()

# skracivanje y tako da ima isti broj varijabli kao i ostale kolone
y = merged['STANJE']

print('samo y, bez ostatka: ')
print(y)

merged = merged.astype({'STAROST':'int'})
merged = merged.astype({'TOAST':'int'})
merged = merged.astype({'ASPECTS':'int'})
merged = merged.astype({'Komorbiditeti':'int'})

# empty cells filled with 0
df_kom = df_kom.fillna(0)

merged = merged.T.drop_duplicates().T

merged = merged.drop(labels='Tip HLP',axis=1)
merged = merged.drop(labels='STANJE',axis=1)

print(merged)