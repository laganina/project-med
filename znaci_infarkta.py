import numpy as np
import pandas as pd

# data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')
data = pd.read_excel(r'C:\Users\Olivera\Documents\PythonScripts\SlogOporavakProjekat2022-23\project-med\podaci.xlsx')

df_infarkt = data[['Strana']]
df_infarkt = pd.DataFrame(df_infarkt)

# LEVO - 1, DESNO - 2, OBE - 3, BEZ ISHEMIJE - 0 
df_infarkt=df_infarkt.replace(to_replace="Levo",value="1")
df_infarkt=df_infarkt.replace(to_replace="Desno",value="2")
# df_infarkt=df_infarkt.replace(to_replace="Desno",value="2")
df_infarkt=df_infarkt.replace(to_replace="Obe (stablo, oba talamusa, više desno)",value="3")
df_infarkt=df_infarkt.replace(to_replace="obe (obe hemisfere cerebeluma, okcipitalno levo)",value="3")
df_infarkt=df_infarkt.replace(to_replace="obe",value="3")
df_infarkt=df_infarkt.replace(to_replace="Obe",value="3")


df_infarkt=df_infarkt.replace(to_replace="Bez ishemije",value="0")
df_infarkt=df_infarkt.replace(to_replace="bez ishemije",value="0")

df_infarkt.rename(columns={"Strana": "Znaci infarkta"}, inplace=True)