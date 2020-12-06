import pandas as pd

lstm = pd.read_csv('../M5Walmart/submission_lstmcnn.csv')
lgbm_valid = pd.read_csv('../M5Walmart/submission_lgbm_053.csv')
lgbm_eval = pd.read_csv('../M5Walmart/submission_eval.csv')

lgbm_valid=lgbm_valid[~lgbm_valid.id.str.contains('evaluation')]

#print(lgbm_valid.shape)


lgbm = pd.concat([lgbm_valid, lgbm_eval])
lgbm =lgbm.set_index('id')
lstm = lstm.set_index('id')
lgbm = lgbm.apply(lambda x: x*0.7)
lstm = lstm.apply(lambda x: x*0.3)
sub_stacking = lgbm.add(lstm)
# cols = [ x for x in sub_stacking.columns][1:]
# sub_stacking = sub_stacking.apply(lambda x: x*0.5)
sub_stacking = sub_stacking.reset_index()
sub_stacking.to_csv("submission_73.csv",index=False)
