import duti, pandas, subprocess, os

subprocess.check_output(['unzip', '-o', 'adult.csv.zip'])

a = pandas.read_csv('./adult.csv') 
for i in a:
    print(set(a[i]))

os.remove('./adult.csv')
