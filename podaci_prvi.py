import pandas as pd
import numpy as np
from tip_cvi import df_tipCVI
from tip_hlp import df_tipHLP
from toast import df_toast
from ct import df_ct
from pol import df_pol
from znaci_infarkta import df_infarkt
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
# data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')
data = pd.read_excel(r'C:\Users\Olivera\Documents\PythonScripts\SlogOporavakProjekat2022-23\project-med\podaci.xlsx')

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
#print('df basic:')
#print(df_basic)

# dimenzije df_basic
#print('*****************************************************************************')
#print('*****************************************************************************')

#----------------------------------------------------------------------------------------------------------------------#
dodatni = data[['TT','Glikemija','MAP','OTT (onset to treatment time)','DNT (door to neadle time)']]

df_dodatni = pd.DataFrame(dodatni)

#print('df_dodatni:')
#print(df_dodatni)
#print('*****************************************************************************')
#print('*****************************************************************************')

#----------------------------------------------------------------------------------------------------------------------#
df_lek = data[['ASA', 'Clopidogrel','OAKT', 'Statini', 'AntiHTA']]
df_lek = pd.DataFrame(df_lek)


#print('df_lek, tek ucitano:')
#print(df_lek)

#----------------------------------------------------------------------------------------------------------------------#
df_kom = data[['AA','HTA','DM','Pušenje','HLP','CMP','Alkohol']]
df_kom = pd.DataFrame(df_kom)
print('Komorbiditeti')
print(df_kom)
#----------------------------------------------------------------------------------------------------------------------#

# ***************************************************
# *********** SPAJANJE DATAFRAME-OVA ****************
# ***************************************************


# spojeni df sa lekovima - BASIC + DODATNI
joined = df_basic.join(df_dodatni, lsuffix='_caller', rsuffix='_other')
#print('df basic + dodatni: ')
#print(joined)

# BASIC + DODATNI + CT
joined = joined.join(df_ct, lsuffix='_caller', rsuffix='_other')
#print('df basic + dodatni + ct: ')
#print(joined)

# BASIC + DODATNI + CT + KOMORBIDITETI
joined = joined.join(df_kom, lsuffix='_caller', rsuffix='_other')    # napravljeno u fajlu ?
#print('df basic + dodatni + ct + komorbiditeti: ')
#print(joined)

# BASIC + DODATNI + CT + KOMORBIDITETI + LEKOVI
joined = joined.join(df_lek, lsuffix='_caller', rsuffix='_other')    # napravljeno u fajlu ? 
#print('df basic + dodatni + ct + komorbiditeti + lekovi: ')
#print(joined)

# BASIC + DODATNI + CT + KOMORBIDITETI + LEKOVI + TIP CVI
joined = joined.join(df_tipCVI, lsuffix='_caller', rsuffix='_other')
#print('df basic + dodatni + ct + komorbiditeti + lekovi + tip_CVI: ')
#print(joined)

# BASIC + DODATNI + CT + KOMORBIDITETI + LEKOVI + TIP CVI + TIP HLP
joined = joined.join(df_tipHLP, lsuffix='_caller', rsuffix='_other')
#print('df basic + dodatni + ct + komorbiditeti + lekovi + tip_CVI + tip_HLP: ')
#print(joined)

# BASIC + DODATNI + CT + KOMORBIDITETI + LEKOVI + TIP CVI + TIP HLP + TOAST
joined = joined.join(df_toast, lsuffix='_caller', rsuffix='_other')
#print('df basic + dodatni + ct + komorbiditeti + lekovi + tip_CVI + tip_HLP + toast: ')
#print(joined)

# BASIC + DODATNI + CT + KOMORBIDITETI + LEKOVI + TIP CVI + TIP HLP + TOAST + POL
joined = joined.join(df_pol, lsuffix='_caller', rsuffix='_other')
#print('df basic + dodatni + ct + komorbiditeti + lekovi + tip_CVI + tip_HLP + pol: ')
#print(joined)

# BASIC + DODATNI + CT + KOMORBIDITETI + LEKOVI + TIP CVI + TIP HLP + TOAST + POL + ZNACI INFARKTA
joined = joined.join(df_infarkt, lsuffix='_caller', rsuffix='_other')
#print('sve spojeno: ')
#print(joined)
#print(f'DIMENZIJE CELOG DF: {joined.size} ')
#print('*****************************************************************************')
#print('*****************************************************************************')

#----------------------------------------------------------------------------------------------------------------------#
# DROPOVACEMO TIP HLP
joined = joined.drop(labels='Tip HLP',axis=1)

# ako su prazni, dodeljuje nan
joined = joined.fillna(np.nan)
joined = joined.replace('nr', np.nan)

# izbacuje sve nanove
joined = joined.dropna()

# resetuje indekse, potrebno zbog izbacenih nanova
joined = joined.reset_index(drop=True)

print('pre izmene lekova i komorbiditeta')
print(joined)
#----------------------------------------------------------------------------------------------------------------------#
#### LEKOVI i KOMORBIDITETI####

# vrednosti od 0 do 5 za da (koji lek) ne
# joined=joined.replace(to_replace="Da",value="1")
# joined=joined.replace(to_replace="Da ",value="1")
# joined=joined.replace(to_replace="da",value="1")
# joined=joined.replace(to_replace="Da (CMP hypertrophica comp)",value="1")
# joined=joined.replace(to_replace="Da (CMP ischaemica9",value="1")
# joined=joined.replace(to_replace="Da (CMP ischaemica)",value="1")
# joined=joined.replace(to_replace="Da (CMP valvulars)",value="1")
# joined=joined.replace(to_replace="Da (CMP valvularis chr. Com EF 45%)",value="1")
# joined=joined.replace(to_replace="Da (CMP valvularis chr. Com)",value="1")
# joined=joined.replace(to_replace="Da (CMP dilatativa, EF 23%)",value="1")
# joined=joined.replace(to_replace="Da (CMP hypertensiva)",value="1")
# joined=joined.replace(to_replace="Da (CMP isch)",value="1")
# joined=joined.replace(to_replace="Da (CMP hypertrophica comp)",value="1")
# joined=joined.replace(to_replace="Da (CMP dilatativa EF 35%)",value="1")
# joined=joined.replace(to_replace="Da (CMP dilatativa)",value="1")
# joined=joined.replace(to_replace="da (CMP hypertensiva hypertrophica comp)",value="1")
# joined=joined.replace(to_replace="da (CMP ischaemica)",value="1")
# joined=joined.replace(to_replace="da, CMP valvularis",value="1")

# joined=joined.replace(to_replace="Ne",value="0")
# joined=joined.replace(to_replace="ne",value="0")

joined=joined.replace(to_replace="Da",value=1)
joined=joined.replace(to_replace="Da ",value=1)
joined=joined.replace(to_replace="da",value=1)
joined=joined.replace(to_replace="Da (CMP hypertrophica comp)",value=1)
joined=joined.replace(to_replace="Da (CMP ischaemica9",value=1)
joined=joined.replace(to_replace="Da (CMP ischaemica)",value=1)
joined=joined.replace(to_replace="Da (CMP valvulars)",value=1)
joined=joined.replace(to_replace="Da (CMP valvularis chr. Com EF 45%)",value=1)
joined=joined.replace(to_replace="Da (CMP valvularis chr. Com)",value=1)
joined=joined.replace(to_replace="Da (CMP dilatativa, EF 23%)",value=1)
joined=joined.replace(to_replace="Da (CMP hypertensiva)",value=1)
joined=joined.replace(to_replace="Da (CMP isch)",value=1)
joined=joined.replace(to_replace="Da (CMP hypertrophica comp)",value=1)
joined=joined.replace(to_replace="Da (CMP dilatativa EF 35%)",value=1)
joined=joined.replace(to_replace="Da (CMP dilatativa)",value=1)
joined=joined.replace(to_replace="da (CMP hypertensiva hypertrophica comp)",value=1)
joined=joined.replace(to_replace="da (CMP ischaemica)",value=1)
joined=joined.replace(to_replace="da, CMP valvularis",value=1)

joined=joined.replace(to_replace="Ne",value=0)
joined=joined.replace(to_replace="ne",value=0)

lekovi = joined[['ASA', 'Clopidogrel', 'OAKT', 'Statini', 'AntiHTA']]
lekovi = pd.DataFrame(lekovi)
# print(lekovi)
# print(lekovi.info())

keys = pd.Series({'ASA': 1, 'Clopidogrel': 2, 'OAKT': 3, 'Statini':4, 'AntiHTA':5})
result  = lekovi.dot(keys)
# print(result)

#----------------------------------------------------------------------------------------------------------------------#
# spajanje i izbacivanje lekova
# joined['Lekovi'] = joined[joined.columns[17:22]].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
joined['Lekovi'] = result 

joined = joined.drop(labels='ASA',axis=1)
joined = joined.drop(labels='Clopidogrel',axis=1)
joined = joined.drop(labels='OAKT',axis=1)
joined = joined.drop(labels='Statini',axis=1)
joined = joined.drop(labels='AntiHTA',axis=1)

#----------------------------------------------------------------------------------------------------------------------#
# spajanje i izbacivanje komorbiditeta
kom = joined[['AA','HTA','DM','Pušenje','HLP','CMP','Alkohol']]
kom = pd.DataFrame(kom)
print('Komorbiditeti')
print(kom)

keys = pd.Series({'AA': 1, 'HTA': 2, 'DM': 3, 'Pušenje':4, 'HLP':5, 'CMP':6, 'Alkohol':7})
result  = kom.dot(keys)

# joined['Komorbiditeti'] = joined[joined.columns[10:17]].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
joined['Komorbiditeti'] = result 


joined = joined.drop(labels='AA',axis=1)
joined = joined.drop(labels='HTA',axis=1)
joined = joined.drop(labels='DM',axis=1)
joined = joined.drop(labels='Pušenje',axis=1)
joined = joined.drop(labels='HLP',axis=1)
joined = joined.drop(labels='CMP',axis=1)
joined = joined.drop(labels='Alkohol',axis=1)

#----------------------------------------------------------------------------------------------------------------------#

#print('index kolona:')
#print(joined.index)
#joined.index.name = 'index'
#print(f'index name: {joined.index.name}')
#print(joined.info())
#print(joined)


#print(f'DIMENZIJE DF POSLE IZBACIVANJA NANOVA, drugi put: {joined.shape} ')
# **********************************************************************
#print('*****************************************************************************')
#print('*****************************************************************************')


# ostali
joined = joined.astype({'ASPECTS' :int})
joined = joined.astype({'TIP CVI' :int})
joined = joined.astype({'CT hiperdenzni znak' :int})
joined = joined.astype({'TOAST' :int})
joined = joined.astype({'OTT (onset to treatment time)' :float})

# DODAJEM PROVERU TIPA 
joined = joined.astype({'POL' :int})
joined = joined.astype({'Znaci infarkta' :int})
# joined = joined.astype({'ASPECTS' :int})



#print(joined.info())
# ********************************************************************************************************
# ********************************************************************************************************
joined_basic = joined[['STAROST', 'NIHSS na prijemu', 'ASPECTS']]
#print('Ista memorija:')
#print(joined_basic is joined)



# **************************************
# *********** LABELA *******************
# **************************************
# AKO PADNE VISE OD 40 POSTO ONDA JE DOSLO DO POBOLJSANJA


# izdvojeni nihss parametri
nihssprijem = joined['NIHSS na prijemu']
#print(f'Dimenzije nihssprijem, {nihssprijem.shape}')
#print('nihssprijem:')
#print(nihssprijem)
#print('*****************************************************************************')
#print('*****************************************************************************')

nihss24 = joined['NIHSS 24h']
#print(f'Dimenzije nihss24, {nihss24.shape}')
#print('nihss24:')
#print(nihss24)
#print('*****************************************************************************')
#print('*****************************************************************************')

# napravljena lista koja ce predstavljati labelu
label = []
# for i in nihssprijem:   # ovo nije dobro
for i in range(len(nihssprijem)):
    label.append((nihssprijem[i] - nihss24[i]) / nihssprijem[i])
#print('label:')
#print(label)
#print(len(label))


# poboljsanje = 0, pogorsanje = 1   # POGORSANJE JE AKO NIJE DOSLO DO POBOLJSANJA, ZNACI NIHA NIJE OPAO ZA VISE OD 40 POSTO 
y = []
for i in range(len(label)):
    if label[i] < 0.4:
        y.append(1)
    else:
        y.append(0)
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


#----------------------------------------------------------------------------------------------------------------------#
# list to data frame
labela = pd.DataFrame(y, columns=['STANJE'])   # DF1 SADRZI STANJE, TJ LABELU
print(labela)
print(labela.info())
broj_jedinica = sum(labela['STANJE'] == 1)
broj_nula = sum(labela['STANJE'] == 0)
print(f'broj_jedinica: {broj_jedinica}')
print(f'broj_nula: {broj_nula}')

print(obelezja.info())
print(obelezja)

# broj_jedinica: 163
# broj_nula: 230

# print('*****************************************************************************')
# print('*****************************************************************************')

# float su:
# STAROST, NIHSS, NIHSS24, TT MASA, GLIKEMIJA, MAP, OTT, DNT


# 


