import pandas as pd

fname = 'Reports/GKAI.html'

table = pd.read_html(fname, header=0, encoding='utf-8', keep_default_na=False)
data = table[0]

#GK weightings
data['Gk1'] = (data['Ref']+data['Agi'])/2
data['Gk2'] = (data['Aer']+data['Cmd']+data['Han']+data['Pos'])/4
data['Gk3'] = (data['Com']+data['Kic']+data['1v1']+data['Thr']+data['Ant']+data['Cnt']+data['Dec'])/7
data['Gk'] = ((data['Gk1']*5)+(data['Gk2']*3)+(data['Gk3']*1))/9
gk = data[['Inf','Name','Position','Age','Gk','xGP/90',]]

#CB
#data['CB1'] = (data['']+data['']+data[''])/3
print(gk.sort_values(by=['Gk'], ascending = False))
