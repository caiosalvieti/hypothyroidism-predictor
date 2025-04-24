# LIBRARIES
import pandas as pd # for data frame operations
import matplotlib.pyplot as plt # for plots
import numpy as np # for math
import seaborn as sns # another way to make plots
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#DATA
dtthyroid = pd.read_csv("/Users/caiosalvieti/Downloads/hypothyroid.data")
dtt = dtthyroid
dtt = dtt.rename(columns={'hypothyroid': 'diseases','72': 'age','M': 'sex','f': 'on_thyroxine','f.1': 'query_on_thyroxine','f.2': 'on_antithyroid_medication','f.3': 'thyroid_surgery','f.4': 'query_hypothyroid','f.5': 'query_hyperthyroid','f.6': 'pregnant','f.7': 'sick','f.8': 'tumor','f.9': 'lithium','f.10': 'goitre','y': 'TSH_measured','30': 'TSH','y.1': 'T3_measured','0.60': 'T3','y.2': 'TT4_measerued','15': 'TT4','y.3': 'T4U_measured','1.48': 'T4U','y.4': 'FTI_measured','10': 'FTI','n': 'TBG_measured','?': 'TBG'})
dtt = dtt.replace("?", np.nan)


#dtt['diseases'] = dtt['diseases'].map({'hypothyroid': 1,'negative': 0})
#dtt['sex'] = dtt['sex'].map({'F': 1,'M': 0})
#dtt['on_thyroxine'] = dtt['on_thyroxine'].map({'f': 0,'t': 1})
#dtt['query_on_thyroxine'] = dtt['query_on_thyroxine'].map({'f': 0,'t': 1})
#dtt['on_antithyroid_medication'] = dtt['on_antithyroid_medication'].map({'f': 0,'t': 1})
#dtt['thyroid_surgery'] = dtt['thyroid_surgery'].map({'f': 0,'t': 1})
#dtt['query_hypothyroid'] = dtt['query_hypothyroid'].map({'f': 0,'t': 1})
#dtt['query_hyperthyroid'] = dtt['query_hyperthyroid'].map({'f': 0,'t': 1})
#dtt['pregnant'] = dtt['pregnant'].map({'f': 0,'t': 1})
#dtt['sick'] = dtt['sick'].map({'f': 0,'t': 1})
#dtt['tumor'] = dtt['tumor'].map({'f': 0,'t': 1})
#dtt['lithium'] = dtt['lithium'].map({'f': 0,'t': 1})
#dtt['goitre'] = dtt['goitre'].map({'f': 0,'t': 1})
#dtt['TSH_measured'] = dtt['TSH_measured'].map({'y': 1,'n': 0})
#dtt['T3_measured'] = dtt['T3_measured'].map({'y': 1,'n': 0})
#dtt['TT4_measerued'] = dtt['TT4_measerued'].map({'y': 1,'n': 0})
#dtt['T4U_measured'] = dtt['T4U_measured'].map({'y': 1,'n': 0})
#dtt['FTI_measured'] = dtt['FTI_measured'].map({'y': 1,'n': 0})
#dtt['TBG_measured'] = dtt['TBG_measured'].map({'y': 1,'n': 0})

# DICIONARY = THE BEST WAY FAST, QUICK AND SHORT WHY NOT?
# KEY:VALUE, KEY:VALUE
# .REPLACE WITH  NEW VARIABLE = TO DTT
dtt = dtt.drop(['TBG_measured','TBG'], axis=1)
dttr = {'hypothyroid': 1, 'negative': 0, 'F': 1, 'M': 0, 'f': 0, 't': 1, 'y': 1, 'n': 0}
dtt = dtt.replace(dttr)
print(dtt)
# Converte a coluna para numérica, forçando erro caso haja valores inválidos
dtt["TSH"] = pd.to_numeric(dtt["TSH"], errors="coerce")

# Converte a coluna "TT4" também para garantir que não há problemas
dtt["TT4"] = pd.to_numeric(dtt["TT4"], errors="coerce")

sns.jointplot(data=dtt, x="TSH", y="TT4", hue="diseases")
plt.show()