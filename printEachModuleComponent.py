import pandas as pd
import numpy as np
from pandas.stats.moments import ewma


def get_year_month(year_month):
    ''' splits the year_month column to year and month and returns them in int format'''
    years = []
    months = []
    for y_m in year_month:
        y,m = y_m.split('/')
        years.append(int(y))
        months.append(int(m))
    return years,months


def get_zero_repairs(st_year,end_year):
    import itertools
    zero_df = []
    years = list(range(st_year,end_year +1))
    months = list(range(1,13))
    zero_df = [(y,m,0) for y,m in itertools.product(years,months)]
    zero_df = pd.DataFrame(zero_df)
    zero_df.columns = ['year','month','number_repair']
    return zero_df


def get_repair_complete(module,component):
    zero_repair = get_zero_repairs(repair_min_year,repair_max_year)
    repair_module_component = repair_train[np.logical_and(
                                repair_train.module_category == module,
                                repair_train.component_category == component)]

    cols_req = ['year_repair','month_repair','number_repair']
    cols_groupby = ['year_repair','month_repair']
    cols_req_final = ['year','month','number_repair']
    repair_train_summ = repair_module_component[cols_req].groupby(cols_groupby,as_index=False).sum()
    repair_train_summ.columns = ['year','month','number_repair']
    repair_merged = pd.merge(zero_repair,repair_train_summ,how='left',on=['year','month'])
    repair_merged['number_repair'] = repair_merged['number_repair_x'] + repair_merged['number_repair_y']

    return repair_merged[cols_req_final]

#Use simple moving average
def simpleMApredict(x,span,periods = pred_period):
    x_predict = np.zeros((span+periods,))
    x_predict[:span] = x[-span:]
    pred =  ewma(x_predict,span)[span:]

    pred = pred.round()
    pred[pred < 0] = 0
    return pred


################################################################################

repair_train = pd.read_csv('Data/RepairTrain.csv')
sale_train = pd.read_csv('Data/SaleTrain.csv')
output_target = pd.read_csv('Data/Output_TargetID_Mapping.csv')
submission = pd.read_csv('Data/SampleSubmission.csv')
pred_period = 19


# Process RepairTrain
#Separate year/month to two columns
repair_train['year_sale2'],repair_train['month_sale2'] = get_year_month(repair_train['year/month(sale)'])
repair_train['year_repair'],repair_train['month_repair'] = get_year_month(repair_train['year/month(repair)'])

repair_min_year = repair_train['year_repair'].min()
repair_max_year = repair_train['year_repair'].max()
sale2_min_year = repair_train['year_sale2'].min()
sale2_max_year = repair_train['year_sale2'].max()

cols_requ = ['module_category','component_category','year_repair','month_repair','number_repair']
cols_groupby = ['module_category','component_category','year_repair','month_repair']
repair_train_summ = repair_train[cols_requ].groupby(cols_groupby).sum()


#Print Total repairNum for each Module Component
cols_requ1 = ['module_category','component_category','number_repair']
cols_groupby1 =  ['module_category','component_category']
all_repairNum_Sum = repair_train[cols_requ1].groupby(cols_groupby1).sum()
all_repairNum_Sum.to_csv('module_component_Sum.csv')

all_repairNum = pd.DataFrame()
print('printing')
#predict for each module and category
for i in range(0,output_target.shape[0],pred_period):
    module = output_target['module_category'][i]
    component = output_target['component_category'][i]

    # missing periods are filled with 0
    X = get_repair_complete(module,component).fillna(0)
    all_repairNum[str(module)+str(component)] = X['number_repair']
    X.to_csv(str(module)+"_"+str(component)+".csv", index=False)

all_repairNum.to_csv('all_repairNum.csv', index=False)
print('done')
