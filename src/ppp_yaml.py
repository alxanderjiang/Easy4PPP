import yaml
import numpy as np
import os
from tqdm import tqdm
import csv
import sys
sys.path.append("src")
from satpos import *
from sppp import *
from sppp_multiGNSS import *


def PPP_YAML_Single(PPP_cfg):
    print("Easy4PPP Configurations:")
    for key in PPP_cfg.keys():
        print(key,PPP_cfg[key])
    
    #单系统双频非差非组合PPP, 解算文件最小配置(观测值/观测值类型/精密星历文件/精密钟差文件/天线文件/结果输出路径)
    obs_file=PPP_cfg['obs_file']                        #观测值文件, RINEX3及以上
    obs_type=PPP_cfg['obs_type']
    if(len(obs_type)!=8):
        obs_type=PPP_cfg['obs_type'][0]                          #兼容多系统数组配置方式
    SP3_file=PPP_cfg['SP3_file']                                 #精密星历文件路径
    CLK_file=PPP_cfg['CLK_file']                                 #精密钟差文件路径
    ATX_file=PPP_cfg['ATX_file']                                 #天线文件, 支持格式转换后的npy文件和IGS天线文件

    out_path=PPP_cfg['out_path']                                 #导航结果输出文件路径
    ion_param=PPP_cfg['ion_param']                               #自定义Klobuchar电离层参数
    
    #可选配置(广播星历文件/DCB改正与产品选项/)
    BRDC_file=PPP_cfg['BRDC_file']                               #广播星历文件, 支持BRDC/RINEX3混合星历
    dcb_correction=PPP_cfg['dcb_correction']                     #DCB修正选项
    dcb_products=PPP_cfg['dcb_products']                         #DCB产品来源, 支持CODE月解文件/CAS日解文件
    dcb_file_0=PPP_cfg['dcb_file_0']                             #频间偏差文件, 支持预先转换后的.npy格式
    dcb_file_1=PPP_cfg['dcb_file_1']
    dcb_file_2=PPP_cfg['dcb_file_2']

    obs_start=PPP_cfg['obs_start']                               #解算初始时刻索引
    obs_epoch=PPP_cfg['obs_epoch']                               #解算总历元数量
    out_age=PPP_cfg['out_age']                                   #最大容忍失锁阈值时间(单位: s, 用于电离层、模糊度状态重置)
    sys_index=PPP_cfg['sys_indexs']                              #此列表控制obsmat读取范围
    sys=["GPS","BDS","GAL"][['G','C','E'].index(sys_index[0])]   #配置解算系统, 支持GPS/BDS/GAL单系统
    dy_mode=PPP_cfg['dy_mode']                                   #PPP动态模式配置, 支持static, dynamic
    try:
        f1=PPP_cfg['freqs'][0][0]                                #配置第一频率
        f2=PPP_cfg['freqs'][0][1]                                #配置第二频率
    except:
        f1=PPP_cfg['freqs'][0]
        f2=PPP_cfg['freqs'][1]
    print(f1,f2)
    el_threthod=PPP_cfg['el_threthod']                           #设置截止高度角
    ex_threshold_v=PPP_cfg['ex_threshold_v']                     #设置先验残差阈值
    ex_threshold_v_sigma=PPP_cfg['ex_threshold_v_sigma']         #设置后验残差阈值
    Mw_threshold=PPP_cfg['Mw_threshold']                         #设置Mw组合周跳检验阈值
    GF_threshold=PPP_cfg['GF_threshold']                         #设置GF组合周跳检验阈值

    sat_out=PPP_cfg['sat_out']                                      #设置排除卫星
    
    #CAS DCB产品数据读取
    if(dcb_correction==1 and dcb_products=='CAS'):
        dcb_file_0,_=CAS_DCB(dcb_file_0,obs_type[0],obs_type[4])
        dcb_file_1=''       #CAS产品同时包含码间和频间偏差
        dcb_file_2=''
    print(dcb_file_0)
    #读取观测文件
    sys_code=['G','C','E','R']
    sys_number=['GPS','BDS','GAL','GLO']
    obs_mat=RINEX3_to_obsmat(obs_file,obs_type,sys=sys_code[sys_number.index(sys)],dcb_correction=dcb_correction,dcb_file_0=dcb_file_0,dcb_file_1=dcb_file_1,dcb_file_2=dcb_file_2)
    obs_mat=reconstruct_obs_mat(obs_mat)
    #删除CAS-DCB产品辅助文件
    if(dcb_file_0!="" and dcb_correction==1):
        os.unlink(dcb_file_0)

    if(not obs_epoch):
        obs_epoch=len(obs_mat)
        print("End index not set, solve all the observations. Total: {}".format(obs_epoch))
    
    #读取精密轨道和钟差文件
    IGS=getsp3(SP3_file)
    clk=getclk(CLK_file)
    
    #读取天线文件
    try:
        #npy格式
        sat_pcos=np.load(ATX_file,allow_pickle=True)
        sat_pcos=eval(str(sat_pcos))
    except:
        #ATX格式
        sat_pcos=RINEX3_to_ATX(ATX_file)
    
    #读取广播星历电离层参数
    if(not len(ion_param)):
        ion_param=RINEX2ion_params(BRDC_file)
    
    #根据配置设置卫星数量
    if(sys=='GPS'):
        sat_num=32
    elif(sys=='BDS'):
        sat_num=65
    elif(sys=='GAL'):
        sat_num=37
    else:
        sat_num=0
    

    #排除精密星历基准卫星
    IGS_PRNS=[list(t.keys())[2:] for t in IGS]
    igs_prns=[]
    for igs_prn in IGS_PRNS:
        if igs_prn not in igs_prns:
            igs_prns.append(igs_prn)
    IGS_PRNS=igs_prns.copy()
    if len(IGS_PRNS)-1:
        for i in range(len(IGS_PRNS)-1):
            t_PRNlists=set(IGS_PRNS[i])&set(IGS_PRNS[i+1])
            IGS_PRNS[i+1]=t_PRNlists.copy()
        IGS_PRNS=t_PRNlists.copy()
    else:
        IGS_PRNS=IGS_PRNS[0]

    for sys in sys_index:
        for prn in range(1,sat_num+1): 
            if("{}{:02d}".format(sys,prn) not in IGS_PRNS):
                sat_out.append("{}{:02d}".format(sys,prn))
    print("Satellites outside for no precise eph",sat_out)
    
    #初始化PPP滤波器
    X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias=init_UCPPP(obs_mat,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=sat_num,f1=f1,f2=f2)
    
    #非差非组合PPP解算
    Out_log=UCPPP(obs_mat,obs_start,obs_epoch,IGS,clk,sat_out,ion_param,sat_pcos,
                  el_threthod=el_threthod,ex_threshold_v=ex_threshold_v,ex_threshold_v_sigma=ex_threshold_v_sigma,
                  Mw_threshold=Mw_threshold,GF_threshold=GF_threshold,dy_mode=dy_mode,
                X=X,Pk=Pk,Qk=Qk,X_time=X_time,phase_bias=phase_bias,GF_sign=GF_sign,Mw_sign=Mw_sign,slip_sign=slip_sign,dN_sign=dN_sign,sat_num=sat_num,out_age=out_age,f1=f1,f2=f2)
    
    #结果以numpy数组格式保存在指定输出目录下, 若输出目录为空, 则存于nav_result
    try:
        np.save(out_path+'/{}.out'.format(os.path.basename(obs_file)),Out_log,allow_pickle=True)
        print("Navigation results saved at ",out_path+'/{}.out'.format(os.path.basename(obs_file)))
    except:
        np.save('nav_result/{}.out'.format(os.path.basename(obs_file)),Out_log,allow_pickle=True)
        print("Navigation results saved at ",'nav_result/{}.out'.format(os.path.basename(obs_file)))
    
    return True


def PPP_YAML_GCE(PPP_cfg):
    #首先可视化配置
    print("Easy4PPP Configurations:")
    for key in PPP_cfg.keys():
        print(key,PPP_cfg[key])
    
    #多系统双频非差非组合PPP, 解算文件最小配置(观测值/观测值类型/精密星历文件/精密钟差文件/天线文件/结果输出路径)
    obs_file=PPP_cfg['obs_file']

    sys_indexs=PPP_cfg['sys_indexs']
    sys_select_num=len(sys_indexs)
    sys_select_ids=sys_indexs.copy()                               #用户选择的原始系统标识
    sys_indexs=['G','C','E']                                       #重置系统标识

    obs_type=PPP_cfg['obs_type']                                   #混合观测值类型, 以RINEX协议为准
    freqs=PPP_cfg['freqs']                                         #各频点观测值中央频率
    
    #系统与信号标识符校验
    try:
        if(len(sys_indexs)==len(obs_type)):
            print("Systems set as: ",sys_indexs)
        else:
            print("Systems set error for: ")
            print("sys_indexs: ", sys_indexs)
            print("obs_type: ",   obs_type)
    except:
        ValueError("Systems not set correctly")

    #重整观测值列表和频率数组
    if(sys_select_num):
        try:
            freqs_G=freqs[sys_select_ids.index("G")]
            obs_type_G=obs_type[sys_select_ids.index("G")]
        except:
            freqs_G=[1575.42E+6, 1227.60E+6]#默认L1/L2
            obs_type_G=['C1C','L1C','D1C','S1C','C2W','L2W','D2W','S2W']
        try:
            freqs_C=freqs[sys_select_ids.index("C")]
            obs_type_C=obs_type[sys_select_ids.index("C")]
        except:
            freqs_C=[1561.098E+6,1268.52E+6]#默认B1/B3
            obs_type_C=['C2I','L2I','D2I','S2I','C6I','L6I','D6I','S6I']
        try:
            freqs_E=freqs[sys_select_ids.index("E")]
            obs_type_E=obs_type[sys_select_ids.index("E")]
        except:
            freqs_E=[1575.42E+6, 1176.45E+6]#默认E1/E5
            obs_type_E=['C1X','L1X','D1X','S1X','C5X','L5X','D5X','S5X']
        freqs=[freqs_G,freqs_C,freqs_E]
        obs_type=[obs_type_G,obs_type_C,obs_type_E]
    
    SP3_file=PPP_cfg['SP3_file']                                #精密星历文件路径
    CLK_file=PPP_cfg['CLK_file']                                #精密钟差文件路径
    ATX_file=PPP_cfg['ATX_file']                                #天线文件, 支持格式转换后的npy文件和IGS天线文件

    out_path=PPP_cfg['out_path']                                        #导航结果输出文件路径
    print("Navigation Results saved in path: ",out_path)
    ion_param=PPP_cfg['ion_param']                                                 #自定义Klobuchar电离层参数
    
    if(len(ion_param)):
        print("ion_params set as: ",ion_param)
    else:
        print("No ion_param, read from file. ")
    
    #可选配置(广播星历文件/DCB改正与产品选项/)
    BRDC_file=PPP_cfg['BRDC_file']                             #广播星历文件, 支持BRDC/RINEX3混合星历
    print("Broadcast eph file: ",BRDC_file)
    dcb_correction=PPP_cfg['dcb_correction']                   #DCB修正选项
    dcb_products=PPP_cfg['dcb_products']                       #DCB产品来源, 支持CODE月解文件/CAS日解文件
    dcb_file_0=PPP_cfg['dcb_file_0']                           #频间偏差文件, 支持预先转换后的.npy格式
    dcb_file_1=PPP_cfg['dcb_file_1']
    dcb_file_2=PPP_cfg['dcb_file_2']

    obs_start=PPP_cfg['obs_start']                             #解算初始时刻索引
    obs_epoch=PPP_cfg['obs_epoch']                             #解算总历元数量
    out_age=PPP_cfg['out_age']                                 #最大容忍失锁阈值时间(单位: s, 用于电离层、模糊度状态重置)
    dy_mode=PPP_cfg['dy_mode']                                 #PPP动态模式配置, 支持static, dynamic
    el_threthod=PPP_cfg['el_threthod']                         #设置截止高度角
    ex_threshold_v=PPP_cfg['ex_threshold_v']                   #设置先验残差阈值
    ex_threshold_v_sigma=PPP_cfg['ex_threshold_v_sigma']       #设置后验残差阈值
    Mw_threshold=PPP_cfg['Mw_threshold']                       #设置Mw组合周跳检验阈值
    GF_threshold=PPP_cfg['GF_threshold']                       #设置GF组合周跳检验阈值
    sat_out=PPP_cfg['sat_out']

    #处理非GRC系统情况:
    if("G" not in sys_select_ids):
        for i in range(1,33):
            sat_out.append("G{:02d}".format(i))
    if("C" not in sys_select_ids):
        for i in range(1,66):
            sat_out.append("C{:02d}".format(i))
    if("E" not in sys_select_ids):
        for i in range(1,37):
            sat_out.append("E{:02d}".format(i))
    
    #
    #多系统观测值分别读取
    STA_name=obs_file.split('/')[-1][:4].upper()
    print("The name of station (RINEX observation format): ",STA_name)
    dcb_file_0_=""
    if "G" in sys_indexs:
        #CAS DCB产品数据读取
        if(dcb_correction==1 and dcb_products=='CAS'):
            dcb_file_0_,_=CAS_DCB_SR(dcb_file_0,obs_type[0][0],obs_type[0][4],STA_name)
            dcb_file_1=''       #CAS产品同时包含码间和频间偏差
            dcb_file_2=''
        obs_mat_GPS=RINEX3_to_obsmat(obs_file,obs_type[0],sys="G",dcb_correction=dcb_correction,dcb_file_0=dcb_file_0_,dcb_file_1=dcb_file_1,dcb_file_2=dcb_file_2)
        obs_mat_GPS=reconstruct_obs_mat(obs_mat_GPS)
        #删除CAS-DCB产品辅助文件
        if(dcb_file_0_!="" and dcb_correction==1):
            os.unlink(dcb_file_0_)
    if "C" in sys_indexs:
        if(dcb_correction==1 and dcb_products=='CAS'):
            dcb_file_0_,_=CAS_DCB_SR(dcb_file_0,obs_type[1][0],obs_type[1][4],STA_name)
            dcb_file_1=''       #CAS产品同时包含码间和频间偏差
            dcb_file_2=''   
        obs_mat_BDS=RINEX3_to_obsmat(obs_file,obs_type[1],sys="C",dcb_correction=dcb_correction,dcb_file_0=dcb_file_0_,dcb_file_1=dcb_file_1,dcb_file_2=dcb_file_2)
        obs_mat_BDS=reconstruct_obs_mat(obs_mat_BDS)
        if(dcb_file_0_!="" and dcb_correction==1):
            os.unlink(dcb_file_0_)
    if "E" in sys_indexs:
        if(dcb_correction==1 and dcb_products=='CAS'):
            dcb_file_0_,_=CAS_DCB_SR(dcb_file_0,obs_type[2][0],obs_type[2][4],STA_name)
            dcb_file_1=''       #CAS产品同时包含码间和频间偏差
            dcb_file_2=''
        obs_mat_GAL=RINEX3_to_obsmat(obs_file,obs_type[2],sys="E",dcb_correction=dcb_correction,dcb_file_0=dcb_file_0_,dcb_file_1=dcb_file_1,dcb_file_2=dcb_file_2)
        obs_mat_GAL=reconstruct_obs_mat(obs_mat_GAL)
        if(dcb_file_0_!="" and dcb_correction==1):
            os.unlink(dcb_file_0_)
    #读取观测文件
    if (not check_obs_mats([obs_mat_GPS,obs_mat_BDS,obs_mat_GAL])):
        ValueError()

    if(not obs_epoch):
        obs_epoch=len(obs_mat_GPS)
        print("End index not set, solve all the observations. Total: {}".format(obs_epoch))
    
    #读取精密轨道和钟差文件
    IGS=getsp3(SP3_file)
    clk=getclk(CLK_file)
    
    #读取天线文件
    try:
        #npy格式
        sat_pcos=np.load(ATX_file,allow_pickle=True)
        sat_pcos=eval(str(sat_pcos))
    except:
        #ATX格式
        sat_pcos=RINEX3_to_ATX(ATX_file)
    
    #读取广播星历电离层参数
    if(not len(ion_param)):
        ion_param=RINEX2ion_params(BRDC_file)
    
    #根据配置设置卫星数量
    sat_num=0
    sat_num_G=0
    sat_num_C=0
    sat_num_E=0
    if('G' in sys_indexs):
        sat_num_G=32
    if('C' in sys_indexs):
        sat_num_C=65
    if('E' in sys_indexs):
        sat_num_E=37
    sat_num=sat_num_G+sat_num_C+sat_num_E
    print("Total satellite number of selected systems: ", sat_num," GPS: ",sat_num_G," BDS: ",sat_num_C," GAL",sat_num_E)
    
    #排除精密星历基准卫星
    IGS_PRNS=[list(t.keys())[2:] for t in IGS]
    igs_prns=[]
    for igs_prn in IGS_PRNS:
        if igs_prn not in igs_prns:
            igs_prns.append(igs_prn)
    IGS_PRNS=igs_prns.copy()
    if len(IGS_PRNS)-1:
        for i in range(len(IGS_PRNS)-1):
            t_PRNlists=set(IGS_PRNS[i])&set(IGS_PRNS[i+1])
            IGS_PRNS[i+1]=t_PRNlists.copy()
        IGS_PRNS=t_PRNlists.copy()
    else:
        IGS_PRNS=IGS_PRNS[0]

    for sys in sys_indexs:
        if(sys=='G'):
            check_num=sat_num_G
        if(sys=='C'):
            check_num=sat_num_C
        if(sys=='E'):
            check_num=sat_num_E
        for prn in range(1,check_num+1): 
            if("{}{:02d}".format(sys,prn) not in IGS_PRNS):
                sat_out.append("{}{:02d}".format(sys,prn))
    print("Satellites outside for no precise eph",sat_out)
    
    # 分别初始化各系统PPP子滤波状态与协方差
    X_G,Pk_G,Qk_G,GF_sign_G,Mw_sign_G,slip_sign_G,dN_sign_G,X_time_G,phase_bias_G=init_UCPPP(obs_mat_GPS,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=sat_num_G,f1=freqs[0][0],f2=freqs[0][1])
    X_C,Pk_C,Qk_C,GF_sign_C,Mw_sign_C,slip_sign_C,dN_sign_C,X_time_C,phase_bias_C=init_UCPPP(obs_mat_BDS,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=sat_num_C,f1=freqs[1][0],f2=freqs[1][1])
    X_E,Pk_E,Qk_E,GF_sign_E,Mw_sign_E,slip_sign_E,dN_sign_E,X_time_E,phase_bias_E=init_UCPPP(obs_mat_GAL,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=sat_num_E,f1=freqs[2][0],f2=freqs[2][1])
    
    #多系统非差PPP滤波状态初始化
    X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias=init_UCPPP_M(X_G,X_C,X_E,
                 Pk_G,Pk_C,Pk_E,
                 Qk_G,Qk_C,Qk_E,
                 GF_sign_G,GF_sign_C,GF_sign_E,
                 Mw_sign_G,Mw_sign_C,Mw_sign_E,
                 slip_sign_G,slip_sign_C,slip_sign_E,
                 dN_sign_G,dN_sign_C,dN_sign_E,
                 X_time_G,X_time_C,X_time_E,
                 phase_bias_G,phase_bias_C,phase_bias_E)

    Out_log=UCPPP_M([obs_mat_GPS,obs_mat_BDS,obs_mat_GAL],obs_start,obs_epoch,IGS,clk,
          sat_out,ion_param,sat_pcos,el_threthod,ex_threshold_v,ex_threshold_v_sigma,Mw_threshold,GF_threshold,dy_mode,
          X,Pk,Qk,phase_bias,X_time,GF_sign,Mw_sign,slip_sign,dN_sign,sat_num,out_age,freqs)

    #结果以numpy数组格式保存在指定输出目录下, 若输出目录为空, 则存于nav_result
    try:
        np.save(out_path+'/{}.out'.format(os.path.basename(obs_file)),Out_log,allow_pickle=True)
        print("Navigation results saved at ",out_path+'/{}.out'.format(os.path.basename(obs_file)))
    except:
        np.save('nav_result/{}.out'.format(os.path.basename(obs_file)),Out_log,allow_pickle=True)
        print("Navigation results saved at ",'nav_result/{}.out'.format(os.path.basename(obs_file)))
    
    return True

def Easy4PPP_YAML(obs_file,SP3_file,CLK_file,ATX_file,Yaml_path,
                  BRDC_file="",
                  out_path='nav_result',sys_indexs=['G','C','E'],
                  obs_type=[['C1C','L1C','D1C','S1C','C2W','L2W','D2W','S2W'],
                            ['C2I','L2I','D2I','S2I','C6I','L6I','D6I','S6I'],
                            ['C1C','L1C','D1C','S1C','C5Q','L5Q','D5Q','S5Q']],
                  freqs=[[1575.42E+6, 1227.60E+6],
                        [1561.098E+6,1268.52E+6],
                        [1575.42E+6, 1176.45E+6]],
                  ion_param=[],
                  dcb_correction=0,
                  dcb_products= 'CAS',
                  dcb_file_0= "data/DCB/CAS0MGXRAP_20241310000_01D_01D_DCB.BSX",
                  dcb_file_1= "",
                  dcb_file_2= "",
                  obs_start=0,
                  obs_epoch=0,
                  out_age= 31,
                  dy_mode= 'static',
                  el_threthod= 15.0,
                  ex_threshold_v= 30,
                  ex_threshold_v_sigma= 4,
                  Mw_threshold= 2.5,
                  GF_threshold= 0.15,
                  sat_out= []):
    #函数: Easy4PPP配置文件快速生成
    #输入: Easy4PPP必要或非必要变量
    #输出: True
    
    STA_name=obs_file.split('/')[-1][:4].upper()
    sys_sign="GCE"
    if(len(sys_indexs))==1:
        sys_sign=sys_indexs[0]
    if(len(sys_indexs)==2):
        if("C" not in sys_indexs):
            sys_sign="GE"
        if("E" not in sys_indexs):
            sys_sign="GC"
    if(len(sys_indexs)==3):
        sys_sign="GCE"
    YAML_name="Easy4PPP_{}_{}.yaml".format(STA_name,sys_sign)
    
    with open(obs_file,"r") as f:
        ls=f.readlines()
        age1='0'
        age2='0'
        for l in ls:
            if("INTERVAL" in l):
                out_age=int(float(l.split()[0]))+1
                age2=0
                age1=0
                break
            if(">" in l and age1=='0'):
                age1=int(l.split()[5])*60+float(l.split()[6])
                continue
            if(">" in l and age2=='0'):
                age2=int(l.split()[5])*60+float(l.split()[6])
                if(age1!=age2):
                    out_age=round(age2-age1)+1
                    break
                else:
                    age2=0
                    break
        if(age1=='0' or age2=='0'):
            out_age=1+1
    lines=[]
    lines.append("## Easy4PPP配置文件")
    lines.append("## Easy4PPP: 纯Python编译的PPP工具箱")
    lines.append("## Configure File of Easy4PPP: An Easily Applied and Recompiled Multi-platform ")
    lines.append("## Precise Point Positioning Toolbox Coded in Python")
    lines.append("")
    lines.append("## 作者: 蒋卓君, 杨泽恩, 黄文静, 钱闯, 武汉理工大学")
    lines.append("## Copyright 2025-, by Zhuojun Jiang, Zeen Yang, Wenjing Huang, Chuang Qian, Wuhan University of Technology, China")          
    lines.append("")
    lines.append("## \"##\" 下为必选配置, \"#\"下为可选配置")
    lines.append("## Configurations below \"##\" are necessary while \"#\" are optional")
    lines.append("")
    lines.append("## 观测文件")
    lines.append("## observation file path")
    lines.append("obs_file: \"{}\"".format(obs_file))
    lines.append("")
    lines.append("## 多系统标识符 (G: GPS C: BDS E: GAL)")
    lines.append("## System Choices (G for GPS, C for BDS, E for GAL)")
    lines.append("sys_indexs: {}".format(str(sys_indexs)))
    lines.append("")
    lines.append("## 多系统频点选择 (以RINEX v3.04协议为准)")
    lines.append("## Multi-GNSS Code&Frequency Choices (RINEX version 3.04 format)")
    lines.append("")
    lines.append("obs_type: {}".format(obs_type))
    lines.append("")
    lines.append("freqs: {}".format(freqs))
    lines.append("")
    lines.append("## 精密产品文件路径")
    lines.append("## Precise Products Path")
    lines.append("SP3_file: \"{}\"".format(SP3_file))               
    lines.append("CLK_file: \"{}\"".format(CLK_file))          
    lines.append("ATX_file: \"{}\"".format(ATX_file))
    lines.append("")
    lines.append("## 导航结果输出文件路径")
    lines.append("## Navigation Results Path")
    lines.append("out_path: \"{}\"".format(out_path)) 
    lines.append("")
    lines.append("## 自定义电离层模型参数 (当前仅支持八参数Klobuchar模型)")
    lines.append("## Ion Model Parameters Set (Klobuchar only in current version)")
    lines.append("ion_param: {}".format(ion_param))
    lines.append("")
    lines.append("# 广播星历文件路径")
    lines.append("# Broadcast Navigation File Path")
    lines.append("BRDC_file: \"{}\"".format(BRDC_file))
    lines.append("")
    lines.append("# DCB修正选项 (0: 不改正DCB 1:改正DCB)")
    lines.append("# DCB Correction (0: off 1:on)") 
    lines.append("dcb_correction: {}".format(dcb_correction))
    lines.append("")
    lines.append("# DCB产品来源 (CODE; CAS)")
    lines.append("# DCB Products (CODE or CAS)")
    lines.append("dcb_products: \'{}\'".format(dcb_products))
    lines.append("")
    lines.append("# DCB产品文件路径(CAS产品同时包括频内和频间偏差)")
    lines.append("# DCB File Path (If 'CAS', dcb_file_1 and dcb_file_2 not available)")
    lines.append("dcb_file_0: \"{}\"".format(dcb_file_0))
    lines.append("dcb_file_1: \"{}\"".format(dcb_file_1))
    lines.append("dcb_file_2: \"{}\"".format(dcb_file_2))
    lines.append("")
    lines.append("# 解算时间段")
    lines.append("# The solution time period") 
    lines.append("obs_start: {}                               #解算初始时刻索引(the beginning epoch of solution)".format(obs_start))
    lines.append("obs_epoch: {}                               #解算总历元数量  (the ending epoch of solution, 0 for total number)".format(obs_epoch))
    lines.append("")
    lines.append("# 最大容忍失锁阈值时间 (单位: s, 用于电离层、模糊度状态重置)")
    lines.append("# The outlier ages of ionospheric delay and ambiguity, expressed in second.")
    lines.append("out_age: {}".format(out_age))
    lines.append("")
    lines.append("# PPP动态模式配置, 支持static, dynamic")
    lines.append("# Dynamic mode, 'static' or 'dynamic'")
    lines.append("dy_mode: '{}'".format(dy_mode))
    lines.append("")
    lines.append("# 卫星截止高度角")
    lines.append("# The threshold of satellite elevation")
    lines.append("el_threthod: {}".format(el_threthod))
    lines.append("")
    lines.append("# 先验残差阈值 (单位: m)")
    lines.append("# The threshold of pre-fit residuals, expressed in meters")
    lines.append("ex_threshold_v: {}".format(ex_threshold_v))
    lines.append("")
    lines.append("# 后验残差阈值")
    lines.append("# The threshold of post-fit residuals, expressed as the multiples of noise residuals (sigma)")
    lines.append("ex_threshold_v_sigma: {}".format(ex_threshold_v_sigma))
    lines.append("# Mw组合周跳检验阈值 (单位: 周)")
    lines.append("# The threshold of phase jump detation for Mw combinations, expressed in cycle")
    lines.append("Mw_threshold: {}".format(Mw_threshold))
    lines.append("")
    lines.append("# GF组合周跳检验阈值 (单位: m)")
    lines.append("# The threshold of phase jump detation for Mw combinations, expressed in meters")
    lines.append("GF_threshold: {}".format(GF_threshold))
    lines.append("")
    lines.append("# 排除卫星PRN码 (不包括因精密星历和钟差基准而排除的卫星)")
    lines.append("# The PRNs of outiler satellites")                                
    lines.append("sat_out: {}".format(sat_out))             
    
    with open(Yaml_path+"/"+YAML_name,"w") as f:
        print("Configurations(.yaml) saved at:" ,Yaml_path+"/"+YAML_name,"out_age=",out_age)
        for line in lines:
            f.write(line+'\n')



# 主函数: python src/ppp_yaml.py {path to your configuration file (.yaml)}
# 可控制台执行, 执行控制台输入的路径中的PPP配置文件
if __name__=='__main__':
    PPP_YAML=sys.argv[1]
    try:
        #读取yaml
        with open(PPP_YAML,"r",encoding="utf-8") as f:
            cfg=yaml.safe_load(f)
        #判断单/多系统分别执行
        if(len(cfg['sys_indexs'])==1):
            print("sys_indexs set as single satellite system")
            PPP_YAML_Single(cfg)
        else:
            print("sys_index set as multi-GNSS")
            PPP_YAML_GCE(cfg)
    except:
        print(PPP_YAML,"Failed")