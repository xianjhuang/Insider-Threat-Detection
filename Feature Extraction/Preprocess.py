# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:23:18 2019
"""
from sklearn import preprocessing
import pandas as pd
import datetime

LOG_FILE = "file.csv"
DEVICE_FILE = "device.csv"
LOGON_FILE = "logon.csv"
EMAIL_FILE = "email.csv"
HTTP_FILE = "http.csv"

def get_logon_features(df_logon):
    full_user_data=[]
    logon_users = df_logon.user.unique().tolist()
    user_date_details = {}
    for user in logon_users:       
        sub_logon_data=df_logon.loc[df_logon['user'] == user,['user','date','activity','pc']]
        sub_logon_data=sub_logon_data.sort_values('date')        
        date_new = []
        date_new = pd.to_datetime(sub_logon_data['date']).dt.date
        sub_logon_data['date_new'] = date_new
        grp_logon_data=sub_logon_data.groupby(['date_new']) 
        for log_date,log_details in grp_logon_data:
            office_start = datetime.datetime(year = log_date.year, month = log_date.month, day = log_date.day, hour = 8, minute = 00, second = 00)
            office_end = datetime.datetime(year = log_date.year, month = log_date.month, day = log_date.day, hour = 19, minute = 00, second = 00)
        
            log_date_list = log_details.date.tolist()
            activity_list = log_details.activity.tolist()
            
            after_start_logon = 0
            after_end_logon = 0
            time_diff_start = 0
            time_diff_end = 0
            L9_session = 0
            for logon_date ,log_activity in zip(log_date_list,activity_list):
                logon_date = datetime.datetime.strptime(logon_date,"%m/%d/%Y %H:%M:%S")
    
                if ((logon_date > office_end) and (log_activity == 'Logon')):
                    time_diff_start = ((logon_date - office_start).total_seconds())/3600
                    after_start_logon = after_start_logon + 1
                    time_diff_end = ((logon_date - office_end).total_seconds())/3600
                    after_end_logon = after_end_logon + 1
                   
                    idx = activity_list.index(log_activity)
                    if(idx < len(activity_list)):
                        if((idx + 1) <= len(activity_list)):
                            if(activity_list[idx + 1] == 'Logoff'):
                                logoff_time = datetime.datetime.strptime(log_date_list[idx + 1],"%m/%d/%Y %H:%M:%S")
                                L9_session = L9_session + (((logon_date - logoff_time).total_seconds())/3600)
                     
            L3_avg_logon_start_time = 0
            if(after_start_logon > 0):
                L3_avg_logon_start_time =   time_diff_start / after_start_logon
                
            L4_avg_logon_end_time = 0
            if(after_end_logon > 0):
                L4_avg_logon_end_time = time_diff_end / after_end_logon
    
            indices = [i for i, x in enumerate(activity_list) if x == "Logon"]       
            L5_num_of_logins = len(indices)
            if (L5_num_of_logins > 1):
                first_login_index = indices[0]
                first_login_time = datetime.datetime.strptime(log_date_list[first_login_index],"%m/%d/%Y %H:%M:%S")
                if(first_login_time > office_start):
                    L1_first_login_diff = ((first_login_time - office_start).total_seconds())/3600#* 24 * 60
                else:
                    L1_first_login_diff = ((office_start - first_login_time).total_seconds())/3600 #* 24 * 60
        
            if(L5_num_of_logins > 1):
                 last_login_index = indices[-1]
                 last_login_time = datetime.datetime.strptime(log_date_list[last_login_index],"%m/%d/%Y %H:%M:%S")
                 if(last_login_time > office_start):
                     L2_last_login_diff = ((last_login_time - office_start).total_seconds())/3600#* 24 * 60
                 else:
                     L2_last_login_diff = ((office_start - last_login_time).total_seconds())/3600 #* 24 * 60
             
            pc_list = log_details.pc.tolist()
            L7_no_of_pc_accessed = len(set(pc_list))
    
            L8_pc_after_office=[]
            L6_session_after_office = 0
            for index in indices:
                if (datetime.datetime.strptime(log_date_list[index],"%m/%d/%Y %H:%M:%S") > office_end):
                    L6_session_after_office = L6_session_after_office + 1
                    L8_pc_after_office.append(pc_list[index])
    
            user_date_details={}
            user_date_details['User'] = user
            user_date_details['Date'] = log_date
            user_date_details['L1'] = L1_first_login_diff
            user_date_details['L2'] = L2_last_login_diff
            user_date_details['L3'] = L3_avg_logon_start_time
            user_date_details['L4'] = L4_avg_logon_end_time
            user_date_details['L5'] = L5_num_of_logins
            user_date_details['L6'] = L6_session_after_office
            user_date_details['L7'] = L7_no_of_pc_accessed
            user_date_details['L8'] = len(L8_pc_after_office)
            user_date_details['L9'] = L9_session
            
            full_user_data.append(user_date_details)     
     
    df_Logon_Features = pd.DataFrame(full_user_data) 
    return df_Logon_Features

def get_device_features(df_device):
    device_users = df_device.user.unique().tolist()
    full_device_list=[]
    for user in device_users:
        sub_device_data=df_device.loc[df_device['user'] == user,['user','date','activity']]
        sub_device_data=sub_device_data.sort_values('date')     
        date_new = []
        date_new = pd.to_datetime(sub_device_data['date']).dt.date
        sub_device_data['date_new'] = date_new
        grp_device_data=sub_device_data.groupby(['date_new']) 
        for device_date,device_details in grp_device_data:
            device_activity_list = device_details.activity.tolist()       
            indices = [i for i, x in enumerate(device_activity_list) if x == "Connect"]
            D2_device_usage_count = len(indices)     
            if D2_device_usage_count > 0:
                device_time_list = device_details.date.tolist()
                office_end = datetime.datetime(year = device_date.year, month = device_date.month, day = device_date.day, hour = 19, minute = 00, second = 00)
                D1_connect_after_office = 0
                for index in indices:
                    if (datetime.datetime.strptime(device_time_list[index],"%m/%d/%Y %H:%M:%S") > office_end):
                        D1_connect_after_office = D1_connect_after_office + 1              
                    
            device_details={}
            device_details['User'] = user
            device_details['Date'] = device_date
            device_details['D1'] = D1_connect_after_office
            device_details['D2'] = D2_device_usage_count
    
            full_device_list.append(device_details)
            
            
    df_Device_Features = pd.DataFrame(full_device_list) 
    return df_Device_Features

def get_file_features(df_file):
    file_users = df_file.user.unique().tolist()
    full_file_list = []
    for user in file_users:
        sub_file_data=df_file.loc[df_file['user'] == user,['user','date','filename']]
        sub_file_data=sub_file_data.sort_values('date')
        date_new = []
        date_new = pd.to_datetime(sub_file_data['date']).dt.date
        sub_file_data['date_new'] = date_new
        grp_file_data=sub_file_data.groupby(['date_new']) 
        for file_date, file_details in grp_file_data:
            file_names = file_details.filename.tolist()
            F1_download_exe_count = 0
            for exe in file_names:
                if (exe.endswith(".exe")):
                    F1_download_exe_count = F1_download_exe_count + 1
            file_details={}
            file_details['User'] = user
            file_details['Date'] = file_date
            file_details['F1'] = F1_download_exe_count
            full_file_list.append(file_details)                 
    df_File_Features = pd.DataFrame(full_file_list) 
    return df_File_Features

def get_email_features(df_email):
    email_users = df_email.user.unique().tolist()
    full_email_list = []
    for email_user in email_users:        
        sub_email_data=df_email.loc[df_email['user'] == email_user,['user','date','to','cc','bcc','size','attachments']]
        sub_email_data=sub_email_data.sort_values('date')
        date_new = []
        date_new = pd.to_datetime(sub_email_data['date']).dt.date
        sub_email_data['date_new'] = date_new
        grp_email_data=sub_email_data.groupby(['date_new'])     
        sender_list =[]
        for email_date, email_details in grp_email_data:            
            sum_size = sub_email_data.loc[sub_email_data['date_new'] == email_date, 'size'].sum()
            to_list = email_details.to.tolist()
            cc_list = email_details.cc.tolist()
            bcc_list = email_details.bcc.tolist()
            email_size = email_details.size       
            E4_avg = 0
            E4_avg = email_size / sum_size
            attach_list = email_details.attachments.tolist()
            attach_list =  [x for x in attach_list if str(x) != 'nan']
            E3_attachment_count = sub_email_data.loc[sub_email_data['date_new'] == email_date, 'attachments'].sum()       
            sender_list = to_list + cc_list + bcc_list
            sender_list = [sender_list for sender in sender_list if str(sender) != 'nan']
            to_list =  [x for x in to_list if str(x) != 'nan']
            cc_list =  [x for x in cc_list if str(x) != 'nan']
            bcc_list = [x for x in bcc_list if str(x) != 'nan']    
            E5_no_of_recepients = 0
            E5_no_of_recepients = len(to_list) + len(cc_list) + len(bcc_list)
            to_list = ';'.join(to_list)
            cc_list = ';'.join(cc_list)
            bcc_list = ';'.join(bcc_list)    
            sender_list = to_list + cc_list + bcc_list
            sender_list = str(to_list).split(";")    
            E1_outside_reciepient_list = 0
            for to_email_id in sender_list:
                if not to_email_id.endswith("@dtaa.com"):
                    E1_outside_reciepient_list = E1_outside_reciepient_list + 1
            email_details={}
            email_details['User'] = email_user
            email_details['Date'] = email_date
            email_details['E1']  = E1_outside_reciepient_list
            email_details['E2']  = 0
            email_details['E3']  = E3_attachment_count
            email_details['E4']  = E4_avg
            email_details['E5']  = E5_no_of_recepients
            
            full_email_list.append(email_details)     
            
    df_Email_Features = pd.DataFrame(full_email_list) 
    return df_Email_Features

def get_http_features(df_http):
    http_users = df_http.user.unique().tolist()
    full_http_list = []
    for http_user in http_users:       
        sub_http_data=df_http.loc[df_http['user'] == http_user,['user','date','url']]
        sub_http_data=sub_http_data.sort_values('date')
        date_new = []
        date_new = pd.to_datetime(sub_http_data['date']).dt.date
        sub_http_data['date_new'] = date_new
        grp_http_data=sub_http_data.groupby(['date_new'])     
        for http_date, http_details in grp_http_data:
            url_list = http_details.url.tolist()
            matching = [s for s in url_list if "wikileaks.org" in s]    
            H1_wikileaks_count = len(matching)            
            http_details={}
            http_details['User'] = http_user
            http_details['Date'] = http_date    
            http_details['H1']  = H1_wikileaks_count
            full_http_list.append(http_details)     
    df_Http_Features = pd.DataFrame(full_http_list) 
    return df_Http_Features

def merge_all_features(df_Logon,df_Email,df_Http,df_File,df_Device):
    new_df = pd.merge(df_Logon, df_Email,  on=['Date','User'] , how = 'outer')    
    new_df = pd.merge(new_df, df_Http,  on=['Date','User'] , how = 'outer')    
    new_df = pd.merge(new_df, df_File,  on=['Date','User'] , how = 'outer')    
    new_df = pd.merge(new_df, df_Device,  on=['Date','User'] , how = 'outer')    
    new_df["L1"].fillna("0", inplace = True) 
    new_df["L2"].fillna("0", inplace = True) 
    new_df["L3"].fillna("0", inplace = True) 
    new_df["L4"].fillna("0", inplace = True) 
    new_df["L5"].fillna("0", inplace = True) 
    new_df["L6"].fillna("0", inplace = True) 
    new_df["L7"].fillna("0", inplace = True) 
    new_df["L8"].fillna("0", inplace = True) 
    new_df["L9"].fillna("0", inplace = True)     
    new_df["E1"].fillna("0", inplace = True) 
    new_df["E2"].fillna("0", inplace = True) 
    new_df["E3"].fillna("0", inplace = True) 
    new_df["E4"].fillna("0", inplace = True) 
    new_df["E5"].fillna("0", inplace = True)         
    new_df["H1"].fillna("0", inplace = True)     
    new_df["F1"].fillna("0", inplace = True) 
    new_df["D1"].fillna("0", inplace = True) 
    new_df["D2"].fillna("0", inplace = True)     
    
    return new_df

def normalize_numeric_features(df_Numeric_Ftrs):
    full_date_list = []
    grp_logon_data=df_Numeric_Ftrs.groupby(['Date']) 
    
    for log_date,log_details in grp_logon_data:
        sub_data=df_Numeric_Ftrs.loc[df_Numeric_Ftrs['Date'] == log_date,['User','L1','L2','L3','L4','L5','L6','L7','L8','L9','E1','E2','E3','E4','E5','H1','F1','D1','D2']]
    
        L1 = sub_data[['L1']].values.astype(float)
        L2 = sub_data[['L2']].values.astype(float)
        L3 = sub_data[['L3']].values.astype(float)
        L4 = sub_data[['L4']].values.astype(float)
        L5 = sub_data[['L5']].values.astype(float)
        L6 = sub_data[['L6']].values.astype(float)
        L7 = sub_data[['L7']].values.astype(float)
        L8 = sub_data[['L8']].values.astype(float)
        L9 = sub_data[['L9']].values.astype(float)
        
        E1 = sub_data[['E1']].values.astype(float)
        E2 = sub_data[['E2']].values.astype(float)
        E3 = sub_data[['E3']].values.astype(float)
        E4 = sub_data[['E4']].values.astype(float)
        E5 = sub_data[['E5']].values.astype(float)
    
        D1 = sub_data[['D1']].values.astype(float)
        D2 = sub_data[['D2']].values.astype(float)
        
        F1 = sub_data[['F1']].values.astype(float)
    
        H1 = sub_data[['H1']].values.astype(float)
    
       # Create a minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()
    
        # Create an object to transform the data to fit minmax processor
        L1_scaled = min_max_scaler.fit_transform(L1)
        L2_scaled = min_max_scaler.fit_transform(L2)
        L3_scaled = min_max_scaler.fit_transform(L3)
        L4_scaled = min_max_scaler.fit_transform(L4)
        L5_scaled = min_max_scaler.fit_transform(L5)
        L6_scaled = min_max_scaler.fit_transform(L6)
        L7_scaled = min_max_scaler.fit_transform(L7)
        L8_scaled = min_max_scaler.fit_transform(L8)
        L9_scaled = min_max_scaler.fit_transform(L9)
        
        E1_scaled = min_max_scaler.fit_transform(E1)
        E2_scaled = min_max_scaler.fit_transform(E2)
        E3_scaled = min_max_scaler.fit_transform(E3)
        E4_scaled = min_max_scaler.fit_transform(E4)
        E5_scaled = min_max_scaler.fit_transform(E5)
    
        D1_scaled = min_max_scaler.fit_transform(D1)
        D2_scaled = min_max_scaler.fit_transform(D2)
    
        F1_scaled = min_max_scaler.fit_transform(F1)
        
        H1_scaled = min_max_scaler.fit_transform(H1)
    
        user = log_details.User.tolist()
    
        L1_List = L1_scaled.tolist()
        L2_List = L2_scaled.tolist()
        L3_List = L3_scaled.tolist()
        L4_List = L4_scaled.tolist()
        L5_List = L5_scaled.tolist()
        L6_List = L6_scaled.tolist()
        L7_List = L7_scaled.tolist()
        L8_List = L8_scaled.tolist()
        L9_List = L9_scaled.tolist()
    
        E1_List = E1_scaled.tolist()
        E2_List = E2_scaled.tolist()
        E3_List = E3_scaled.tolist()
        E4_List = E4_scaled.tolist()
        E5_List = E5_scaled.tolist()
    
        D1_List = D1_scaled.tolist()
        D2_List = D2_scaled.tolist()
    
        F1_List = F1_scaled.tolist()
        H1_List = H1_scaled.tolist()
    
        # Run the normalizer on the dataframe
        df_normalized = {}
        df_normalized_full = []
    
        for i, L1 in enumerate(L1_List):
            df_normalized={}
            df_normalized['Date'] = log_date
            df_normalized['User'] = user[i]
            df_normalized['L1_norm'] = (L1_List[i])[0]
            df_normalized['L2_norm'] = (L2_List[i])[0]
            df_normalized['L3_norm'] = (L3_List[i])[0]
            df_normalized['L4_norm'] = (L4_List[i])[0]
            df_normalized['L5_norm'] = (L5_List[i])[0]
            df_normalized['L6_norm'] = (L6_List[i])[0]
            df_normalized['L7_norm'] = (L7_List[i])[0]
            df_normalized['L8_norm'] = (L8_List[i])[0]
            df_normalized['L9_norm'] = (L9_List[i])[0]
            
            df_normalized['E1_norm'] = (E1_List[i])[0]
            df_normalized['E2_norm'] = (E2_List[i])[0]
            df_normalized['E3_norm'] = (E3_List[i])[0]
            df_normalized['E4_norm'] = (E4_List[i])[0]
            df_normalized['E5_norm'] = (E5_List[i])[0]
    
            df_normalized['F1_norm'] = (F1_List[i])[0]
            df_normalized['H1_norm'] = (H1_List[i])[0]
    
            df_normalized['D1_norm'] = (D1_List[i])[0]
            df_normalized['D2_norm'] = (D2_List[i])[0]
    
            df_normalized_full.append(df_normalized)
           
        for index in range(len(df_normalized_full)):
            full_date_list.append(df_normalized_full[index])    
    
    df_Norm = pd.DataFrame(full_date_list) 
    df_Norm.set_index(['Date'], inplace=True)
    return df_Norm

def assign_class_label(features):
    answers = pd.read_csv("insiders.csv",usecols= ['scenario', 'user','start','end'])   
    mal_date  = pd.read_csv("malicious_instances.csv",usecols= ['date','user'])   
    
    insiders = answers.user.unique().tolist()   
    for index,row in features.iterrows():    
        if(row['User'] in insiders):
            sub_mal_inst = mal_date.loc[mal_date['user'] == row['User'],['date','user']]    
            sub_logon_data=answers.loc[answers['user'] == row['User'],['scenario','start','end']]
            start_date_ans = pd.to_datetime(sub_logon_data['start']).dt.date
            end_date_ans = pd.to_datetime(sub_logon_data['end']).dt.date
            sub_mal_inst['date'] = pd.to_datetime(sub_mal_inst['date']).dt.date 
            mal_inst_datelist = sub_mal_inst.date.tolist()
            mal_dt_format=[]
            format_str = '%Y-%m-%d'
            for dat_ls in mal_inst_datelist:
                dte_log = datetime.datetime(year = dat_ls.year, month = dat_ls.month, day = dat_ls.day)      
                datetime_obj = datetime.datetime.strftime(dte_log, format_str)
                mal_dt_format.append(datetime_obj)                                           
            log_date=row['Date']   
            str_start_dt = start_date_ans.to_string().split()
            str_end_dt = end_date_ans.to_string().split()     
            log_date = datetime.datetime.strptime(log_date,"%Y-%m-%d")
            if(row['Date'] not in mal_dt_format):
                features.loc[index,'Class_Label'] = 0
            else:   
                if(log_date >= datetime.datetime.strptime(str_start_dt[1],"%Y-%m-%d")) and (log_date <= datetime.datetime.strptime(str_end_dt[1],"%Y-%m-%d")):                         
                    features.loc[index,'Class_Label'] = 1    
                else:
                    features.loc[index,'Class_Label'] = 0  
        else:       
            features.loc[index,'Class_Label'] = 0
    return features
    


def main():
    
    df_File = pd.read_csv(LOG_FILE,usecols= ['date', 'user','filename'])
    df_Device = pd.read_csv(DEVICE_FILE,usecols= ['date', 'user','pc','activity'])
    df_Logon = pd.read_csv(LOGON_FILE,usecols= ['date', 'user','pc','activity'])
    df_Email = pd.read_csv(EMAIL_FILE,usecols= ['date', 'user','to','cc','bcc','size','attachments'])
    df_Http = pd.read_csv(HTTP_FILE,usecols= ['date', 'user','url'])
    
    df_Log_Ftrs = get_logon_features(df_Logon)
    df_Device_Ftrs = get_device_features(df_Device)
    df_File_Frs = get_file_features(df_File)
    df_Email_Ftrs = get_email_features(df_Email)
    df_Http_Ftrs = get_http_features(df_Http)
    
    df_Numeric_Ftrs = merge_all_features(df_Log_Ftrs,df_Device_Ftrs,df_File_Frs,df_Email_Ftrs,df_Http_Ftrs)
    df_Normalized = normalize_numeric_features(df_Numeric_Ftrs)
    
    df_Labelled_Data = assign_class_label(df_Normalized)
    df_Labelled_Data.to_csv("Features.csv")     
    
    print("\n\n Processing Done !!!!!")
