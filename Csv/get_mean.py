import pandas as pd
import sys

if __name__ == '__main__':
	df = pd.read_csv(sys.argv[1], sep = ',', header = None)
	mean_time = sum(df.iloc[:,0].values)/len(df.iloc[:,0])
	mean_cost = sum(df.iloc[:,1].values)/len(df.iloc[:,1])
	
	dfnew = pd.DataFrame([['mean time', 'mean cost'], [mean_time, mean_cost]])
	df = df.append(dfnew, ignore_index=True)
	print(df)
	df.to_csv(sys.argv[1], header = False, index = False)
	# df.to_csv('new.csv', header = False, index = False)
