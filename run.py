import os

#for i in range(10):
#	os.system("python dl_perdict.py --data_path data/OQMD/data.csv --result_name OQMD_data_onehot{}.csv --fea_type one_hot_vec".format(str(i)))
#for i in range(10):
#    os.system("python dl_perdict.py --data_path data/OQMD/data.csv --result_name OQMD_data_magpie{}.csv --fea_type magpie".format(str(i)))

for i in range(10):
    os.system("python dl_perdict.py --data_path data/OQMD/data.csv --result_name OQMD_data_atom2vec{}.csv --fea_type atom2vec".format(str(i)))

# for i in range(10):
# 	os.system("python dl_perdict.py --data_path data/superconder/supercon-tc.csv --result_name Superconder_data_magpie{}.csv --fea_type magpie".format(str(i)))

