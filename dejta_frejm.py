import tensorflow
import keras
import pandas as pd
import numpy as np

data = pd.read_excel(
    r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\podaci.xlsx')

# osnovna obelezja: starost, nihss na prijemu, aspect score

# data frame sa osnovnim obelezjima
basic = data[['STAROST', 'NIHSS na prijemu', 'ASPECTS', 'NIHSS 24h']]
df_basic = pd.DataFrame(basic)

# df sa dodatnim obelezjima
dodatni = data[['CT hiperdenzni znak','TT','Glikemija','MAP','OTT (onset to treatment time)','DNT (door to neadle time)'
                ,'TIP CVI','TOAST']]

df_dodatni = pd.DataFrame(dodatni)


#da ne bilo koja pretvoreno u 1 0 2

df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Bilo koja', 2)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Da', 1)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Ne', 0)

# tip cvi - pretvoreni u 0 1 2 3

df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI', 0)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('PACI', 1)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI', 2)
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
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Stroke mimic', 5)

# spojeni df

joined = df_basic.join(df_dodatni, lsuffix='_caller', rsuffix='_other')

# joined df sa izbacenim vrstama koje sadrze NaN value

joined = joined.apply(pd.to_numeric, errors='coerce')
joined = joined.dropna()

joined = joined.dropna().reset_index(drop=True)




'''

koriscena terapija - od asa do antihta sva obelezja, 
komorbiditeti/faktori rizika - od hta do alkohola pri cemu dodajemo jos jednu vrednost obelezja NE nema faktora rizika

ne izbacujemo u prvoj verziji nista
(terapija) lekove grupisati u jedno obelezje i dati im vrednosti od 0 do 5 sa vrednoscu da nije nista uzimao pacijent, napraviti legendu pored
(komorbititeti) grupisemo u jedno obelezje i kodiramo sa vrednostima od 0 do 7 i dodatna da nema nikakvog komorbiditeta
'''

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
merged = pd.concat([df, df1], axis=1)

print(merged)
