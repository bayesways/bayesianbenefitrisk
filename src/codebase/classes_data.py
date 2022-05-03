from codebase.data import get_final_dataset, interweave_df
from codebase.file_utils import save_obj, load_obj
from codebase.data import create_group_data
from pdb import set_trace


class Data:
    def __init__(self, name):
        self.name = name

    def generate(self, sim_case=0, interweave=True):
        if sim_case==0:
            df = get_final_dataset()
            # df = df.sample(frac=1, random_state=1)
            indx_order = load_obj('index_order', "../dat/")
            df = df.loc[indx_order]
            if interweave:
                df = interweave_df(df)
            else:
               pass
            tmp_dct = dict()
            y = df.iloc[:, 2:].values
            df["atrtgrp"] = df["atrtgrp"].replace({"AVM": 1, "MET": 2, "RSG": 3})
            tmp_dct["grp"] = df["atrtgrp"].values.astype(int)
            # tmp_dct["K"] = 2
            # tmp_dct["J"] = 6
            tmp_dct["Kc"] = 2
            tmp_dct["Kb"] = 4
            tmp_dct["y"] = y
            tmp_dct['yc']= y[:, :2]
            tmp_dct['yb'] = y[:, 2:].astype(int)
            tmp_dct["N"] = df.shape[0]
            tmp_dct["number_of_groups"] = 3
            tmp_dct['stan_constants'] = ['N', 'Kb', 'Kc', 'number_of_groups']
            tmp_dct['stan_data'] = ['yb', 'yc', 'grp']
            self.raw_data = tmp_dct
            self.size = tmp_dct['N']
        elif sim_case==1:
            group_data = create_group_data(
                200,
                corr=1,
                rho=0,
                diff_grps=0,
                seed1=0,
                seed2=6
            )
            tmp_dct = dict()
            tmp_dct["grp"] = group_data["grp"].astype(int)
            tmp_dct["Kc"] = 2
            tmp_dct["Kb"] = 4
            tmp_dct["y"] = group_data['y']
            tmp_dct['yc']= group_data['yc']
            tmp_dct['yb'] = group_data['yb'].astype(int)
            tmp_dct["N"] = group_data['N']
            tmp_dct["number_of_groups"] = 2
            tmp_dct['stan_constants'] = ['N', 'Kb', 'Kc', 'number_of_groups']
            tmp_dct['stan_data'] = ['yb', 'yc', 'grp']
            self.raw_data = tmp_dct
            self.size = tmp_dct['N']
        else:
            print('\n\nsim_case not found')
            return None

    def save_data(self):
        save_obj(self.raw_data, self.name, '../dat/')

    def load_data(self):
        group_data = load_obj(self.name, '../dat/') 
        self.raw_data = group_data  
        self.size = group_data['N'] 


    def get_stan_data(self):
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name]
        return stan_data

    def get_stan_data_at_t(self, t):
        assert t<=self.size
        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        stan_data['N'] = 1
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name][t]
        return stan_data

    def get_stan_data_at_t2(self, t):
        assert t<=self.size

        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        stan_data['N'] = 1
        for name in ['yb','yc']:
            stan_data[name] = self.raw_data[name][t].reshape(1,-1)
        stan_data['grp'] = self.raw_data['grp'][t].reshape(1,)
        return stan_data

    def get_stan_data_upto_t(self, t):
        assert t<=self.size

        stan_data = dict()
        for name in self.raw_data['stan_constants']:
            stan_data[name] = self.raw_data[name]
        stan_data['N'] = t
        for name in self.raw_data['stan_data']:
            stan_data[name] = self.raw_data[name][:t]
        return stan_data

    def keep_data_after_t(self, t):
        assert t<self.size
        self.raw_data["grp"] = self.raw_data["grp"][t:].astype(int).copy()
        self.raw_data["y"] = self.raw_data["y"][t:].copy()
        self.raw_data["yb"] = self.raw_data["yb"][t:].astype(int).copy()
        self.raw_data["yc"] = self.raw_data["yc"][t:].copy()
        self.raw_data["N"] = self.raw_data["N"]-t
        self.size = self.raw_data["y"].shape[0]
