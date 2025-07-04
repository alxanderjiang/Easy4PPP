import numpy as np
import os
from tqdm import tqdm
import csv
from satpos import *
from sppp import *

def reconstruct_obs_mat(obs_mat):
    #函数: 重整有效观测数据字典(根据Epoch_OK标识)
    #输入: 有效观测数据字典
    #输出: 重整后的观测数据字典
    r_obsmat=[]
    for i in range(len(obs_mat)):
        if(obs_mat[i][0]['Epoch_OK'])!=0:
            continue
        else:
            r_obsmat.append(obs_mat[i])
    #返回观测数据字典
    return r_obsmat

def check_obs_mats(obs_mats):
    #函数: 校验多系统观测值匹配性
    #输入: 多系统观测值列表
    #输出: 观测值数据
    lens=[]
    for i in range(len(obs_mats)):
        lens.append(len(obs_mats[i]))
    #列表仅单值
    if(lens==1):
        print("Only one system observed")
        return True
    #列表有多系统观测
    for i in range(len(lens)-1):
        if(lens[i]!=lens[i+1]):
            print("Observations among systems not equal")
            return False
    #长度校验通过, 开始时间校验
    for i in range(len(obs_mats[0])):
        GPS_week=obs_mats[0][i][0]["GPSweek"]
        GPS_sec=obs_mats[0][i][0]["GPSsec"]
        for j in range(len(obs_mats)):
            check_week=obs_mats[j][i][0]['GPSweek']
            check_sec=obs_mats[j][i][0]['GPSsec']
            if(check_week!=GPS_week or check_sec!=GPS_sec):
                print("Observations among systems not in the same time period")
                return False
    
    #全部校验通过, 返回True
    return True

def CAS_DCB_SR(filename,osignal='C1W',tsignal='C2W',sta=""):
    #读取卫星DCB
    dcb_file_0,DCB_mat=CAS_DCB(filename,osignal,tsignal)
    with open(filename,"r") as f:
        lines=f.readlines()
        header=0
        cbias_receiver=0.0
        for line in lines:
            if("+BIAS/SOLUTION" in line):
                header=1
                continue
            if("-BIAS/SOLUTION" in line):
                break

            if(header==1):
                #目标DCB
                #Target DCB
                if(osignal in line and tsignal in line):
                    ls=line.split()
                    if(len(ls[3])==4):
                        if(sta==ls[3]):
                            #print(ls)
                            cbias_receiver=float(ls[9])*1e-9*satpos.clight
                            break
                        # PRN=ls[2]
                        # cbias[PRN]=[osignal+'_'+tsignal,float(ls[8])*1e-9*satpos.clight]
    #读取到测站DCB, 叠加到卫星DCB中
    if(cbias_receiver!=0.0):
        print("Receiver DCB of {}: {}->{} {}m".format(sta,osignal,tsignal,cbias_receiver))
    for key in DCB_mat.keys():
        DCB_mat[key][1]=DCB_mat[key][1]+cbias_receiver
    np.save("{}_{}.npy".format(osignal,tsignal),DCB_mat)
    return "{}_{}.npy".format(osignal,tsignal),DCB_mat

#多系统非差非组合PPP状态合并初始化
def init_UCPPP_M(X_G,X_C,X_E,
                 Pk_G,Pk_C,Pk_E,
                 Qk_G,Qk_C,Qk_E,
                 GF_sign_G,GF_sign_C,GF_sign_E,
                 Mw_sign_G,Mw_sign_C,Mw_sign_E,
                 slip_sign_G,slip_sign_C,slip_sign_E,
                 dN_sign_G,dN_sign_C,dN_sign_E,
                 X_time_G,X_time_C,X_time_E,
                 phase_bias_G,phase_bias_C,phase_bias_E):
    #函数: 多系统非差非组合PPP状态合并初始化
    #输入: sppp.py中单系统初始化结果
    #输出: sppp_multiGNSS.py所需PPP输入
    #各系统星座卫星数量
    sat_num_G=int((X_G.shape[0]-5)/3)
    sat_num_C=int((X_C.shape[0]-5)/3)
    sat_num_E=int((X_E.shape[0]-5)/3)
    #状态、方差、过程噪声向量矩阵生成
    X=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,1))#将ISB置于状态向量最末尾
    X_time=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,1))
    Pk=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,5+3*(sat_num_G+sat_num_C+sat_num_E)+2),dtype=np.float64)#滤波方差阵生成
    Qk=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,5+3*(sat_num_G+sat_num_C+sat_num_E)+2),dtype=np.float64)#滤波过程噪声阵生成
    
    X[0][0]=X_G[0][0]                                    #以GPS系统位置、钟差、ZWD为初始状态
    X[1][0]=X_G[1][0]
    X[2][0]=X_G[2][0]
    X[3][0]=X_G[3][0]
    X[4][0]=X_G[4][0]
    for i in range(5):
        Pk[i][i]=Pk_G[i][i]
        Qk[i][i]=Qk_G[i][i]
        X_time[i][0]=X_time_G[i][0]
    
    for i in range(0,3*sat_num_G):                       #GPS电离层、模糊度
        X[5+i][0]=X_G[5+i][0]
        X_time[5+i][0]=X_time_G[5+i][0]
        Pk[5+i][5+i]=Pk_G[5+i][5+i]
        Qk[5+i][5+i]=Qk_G[5+i][5+i]
    for i in range(0,3*sat_num_C):                       #BDS电离层、模糊度
        X[5+3*sat_num_G+i][0]=X_C[5+i][0]
        X_time[5+3*sat_num_G+i][0]=X_time_C[5+i][0]
        Pk[5+3*sat_num_G+i][5+3*sat_num_G+i]=Pk_C[5+i][5+i]                
        Qk[5+3*sat_num_G+i][5+3*sat_num_G+i]=Qk_C[5+i][5+i]                
    for i in range(0,3*sat_num_E):                       #GAL电离层、模糊度
        X[5+3*sat_num_G+3*sat_num_C+i][0]=X_E[5+i][0]
        X_time[5+3*sat_num_G+3*sat_num_C+i][0]=X_time_E[5+i][0]
        Pk[5+3*sat_num_G+3*sat_num_C+i][5+3*sat_num_G+3*sat_num_C+i]=Pk_E[5+i][5+i]
        Qk[5+3*sat_num_G+3*sat_num_C+i][5+3*sat_num_G+3*sat_num_C+i]=Qk_E[5+i][5+i]
           
    #ISB区块合并
    X[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][0]=X_C[3][0]-X_G[3][0]        #ISB_BDS
    X_time[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][0]=X_time[3][0]          
    Pk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][5+3*sat_num_G+3*sat_num_C+3*sat_num_E]=Pk[3][3]
    Qk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][5+3*sat_num_G+3*sat_num_C+3*sat_num_E]=Qk[3][3]
    X[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][0]=X_E[3][0]-X_G[3][0]      #ISB_GAL
    X_time[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][0]=X_time[3][0]
    Pk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1]=Pk[3][3]
    Qk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1]=Qk[3][3]

    #历元间差分检验量合并
    GF_sign=np.concatenate((GF_sign_G,GF_sign_C,GF_sign_E))
    Mw_sign=np.concatenate((Mw_sign_G,Mw_sign_C,Mw_sign_E))
    slip_sign=np.concatenate((slip_sign_G,slip_sign_C,slip_sign_E))
    dN_sign=np.concatenate((dN_sign_G,dN_sign_C,dN_sign_E))

    #相位误差字典合并
    phase_bias={}
    for key in phase_bias_G.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_G[key]
    for key in phase_bias_C.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_C[key]
    for key in phase_bias_E.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_E[key]
    
    return X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias

#多系统PPP观测模型构建
def createKF_HRZ_M(obslist,rt_unix,X,X_time,Pk,Qk,ion_param,phase_bias,peph_sat_pos,freqs,ex_threshold_v=30,exthreshold_v_sigma=4,post=True):
    
    #初始化卫星数量
    sat_num=len(obslist)
    sys_sat_sum=round((X.shape[0]-7)/3)#GCE三系统
    sat_out=[]
    sat_out_post=[]
    t_phase_bias=phase_bias.copy()
    #光速, GPS系统维持的地球自转角速度(弧度制)
    clight=2.99792458e8
    OMGE=7.2921151467E-5
    #dt=0.001    #计算卫星速度用于相对论效应改正(改正到钟差, 由SPP_from_IGS完成)
    rr=np.array([X[0],X[1],X[2],X[3]]).reshape(4)

    dr=solid_tides(rt_unix,X)

    rr[0]=rr[0]+dr[0]
    rr[1]=rr[1]+dr[1]
    rr[2]=rr[2]+dr[2]
    
    #创建设计矩阵和观测值矩阵(观测模型)
    H=np.zeros((4*sat_num,3*sat_num+7),dtype=np.float64)
    #Z=np.zeros((4*sat_num,1),dtype=np.float64)
    R=np.eye(4*sat_num,dtype=np.float64)
    v=np.zeros((4*sat_num,1),dtype=np.float64)
    #print("H,Z,R",H.shape,Z.shape,R.shape)
    for i in range(sat_num): #逐卫星按行创建设计矩阵
        #状态索引求解
        si_PRN=obslist[i]['PRN']
        sys_shift=0
        f1=freqs[0][0]
        f2=freqs[0][1]
        sys_sat_num=32      #GPS系统星座卫星数量
        if('C' in si_PRN):
            sys_shift=32   #多系统索引偏置(GPS)
            sys_sat_num=65
            f1=freqs[1][0]
            f2=freqs[1][1]
        if('E' in si_PRN):
            sys_shift=32+65   #多系统索引偏置(GPS+BDS)
            sys_sat_num=37
            f1=freqs[2][0]
            f2=freqs[2][1] 
        PRN_index=int(si_PRN[1:])-1
        ion_index=5+3*sys_shift+PRN_index
        N1_index=5+3*sys_shift+sys_sat_num+PRN_index
        N2_index=5+3*sys_shift+sys_sat_num*2+PRN_index
        #观测时间&观测值
        rt_unix=rt_unix
        ##伪距&相位&CNo
        p1=obslist[i]['OBS'][0]
        l1=obslist[i]['OBS'][1]
        s1=obslist[i]['OBS'][4]
        p2=obslist[i]['OBS'][5]
        l2=obslist[i]['OBS'][6]
        s2=obslist[i]['OBS'][9]
        #print(p1,p2,phi1,phi2)
        
        #卫星位置
        si_PRN=obslist[i]['PRN']
        rs=[peph_sat_pos[si_PRN][0],peph_sat_pos[si_PRN][1],peph_sat_pos[si_PRN][2]]
        dts=peph_sat_pos[si_PRN][3]
        drs=[peph_sat_pos[si_PRN][4],peph_sat_pos[si_PRN][5],peph_sat_pos[si_PRN][6]]

        #线性化的站星向量
        r0=sqrt( (rs[0]-rr[0])*(rs[0]-rr[0])+(rs[1]-rr[1])*(rs[1]-rr[1])+(rs[2]-rr[2])*(rs[2]-rr[2]) )

        #线性化的站星单位向量
        urs_x=(rr[0]-rs[0])/r0
        urs_y=(rr[1]-rs[1])/r0
        urs_z=(rr[2]-rs[2])/r0

        #对流层延迟投影函数
        Mh,Mw=NMF(rr,rs,rt_unix)
        #电离层延迟投影函数
        Mi=IMF_ion(rr,rs)

        #单卫星四行设计矩阵分量构建
        #p1
        H_sub1=np.zeros(3*sat_num+7,dtype=np.float64)   #初始化频1伪距行
        H_sub1[0:5]=[urs_x,urs_y,urs_z,1,Mw]            #基础项
        H_sub1[5+i]=1                                   #频1伪距STEC系数
        if("C" in si_PRN):
            H_sub1[-2]=1                                #ISB_BDS
        if("E" in si_PRN):
            H_sub1[-1]=1                                #ISB_GAL
        #l1
        H_sub2=np.zeros(3*sat_num+7,dtype=np.float64)   #初始化频1相位行
        H_sub2[0:5]=[urs_x,urs_y,urs_z,1,Mw]            #基础项
        H_sub2[5+i]=-1                                  #频1相位STEC系数
        H_sub2[5+sat_num+i]=1                           #频1模糊度
        if("C" in si_PRN):
            H_sub2[-2]=1                                #ISB_BDS
        if("E" in si_PRN):
            H_sub2[-1]=1                                #ISB_GAL
        #p2
        H_sub3=np.zeros(3*sat_num+7,dtype=np.float64)   #初始化频2伪距行
        H_sub3[0:5]=[urs_x,urs_y,urs_z,1,Mw]            #基础项
        H_sub3[5+i]=f1*f1/f2/f2                         #频2伪距STEC系数
        if("C" in si_PRN):
            H_sub3[-2]=1                                #ISB_BDS
        if("E" in si_PRN):
            H_sub3[-1]=1                                #ISB_GAL
        #l2
        H_sub4=np.zeros(3*sat_num+7,dtype=np.float64)   #初始化频1相位行
        H_sub4[0:5]=[urs_x,urs_y,urs_z,1,Mw]            #基础项
        H_sub4[5+i]=-f1*f1/f2/f2                        #频2相位STEC系数
        H_sub4[5+2*sat_num+i]=1                         #频2模糊度
        if("C" in si_PRN):
            H_sub4[-2]=1                                #ISB_BDS
        if("E" in si_PRN):
            H_sub4[-1]=1                                #ISB_GAL

        #设计矩阵
        H[i*4]=H_sub1
        H[i*4+1]=H_sub2
        H[i*4+2]=H_sub3
        H[i*4+3]=H_sub4

        #相位改正
        phw=sat_phw(rt_unix+rr[3]/clight,si_PRN,1,rr,rs,drs,t_phase_bias)
        l1=l1-phw
        l2=l2-phw
        t_phase_bias[si_PRN]={}
        t_phase_bias[si_PRN]['phw']=phw
        
        #伪距自转改正
        r0=r0+OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/clight
        
        #残差向量
        isb=0.0
        if("C" in si_PRN):
            isb=X[-2][0]
        if("E" in si_PRN):
            isb=X[-1][0]

        v[i*4]=  p1 -           (r0 + rr[3]+ isb - (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + (1*X[ion_index][0]) )
        v[i*4+1]=l1*clight/f1 - (r0 + rr[3]+ isb - (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( -1*X[ion_index][0]) + (1*X[N1_index][0]) )
        v[i*4+2]=p2 -           (r0 + rr[3]+ isb - (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( f1*f1/f2/f2*X[ion_index][0]) )
        v[i*4+3]=l2*clight/f2 - (r0 + rr[3]+ isb - (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( -f1*f1/f2/f2*X[ion_index][0]) + (1*X[N2_index][0]) )
        #观测噪声(随机模型)
        
        _,el=satpos.getazel(rs,rr)
        var=0.003*0.003+0.003*0.003/sin(el)/sin(el)
        
        # var=0.00224*10**(-s1 / 10)
        # var_1=1.0
        # var_2=1.0
        var_ion=Qk[ion_index][ion_index]
        var_N1=Qk[N1_index][N1_index]
        var_N2=Qk[N2_index][N2_index]
        var_trop=0.01*0.01
        var_ion=0.0
        var_N1=0.0
        var_N2=0.0
        
        R[i*4][i*4]=100*100*(var)+var_ion+var_trop#伪距/相位标准差倍数
        R[i*4+1][i*4+1]=var+var_ion+var_N1+var_trop
        R[i*4+2][i*4+2]=100*100*(var)+var_ion+var_trop
        R[i*4+3][i*4+3]=var+var_ion+var_N2+var_trop
        
        #验前残差粗差识别
        if(post==False):
            if(abs(v[i*4])>ex_threshold_v or abs(v[i*4+1])>ex_threshold_v or abs(v[i*4+2])>ex_threshold_v or abs(v[i*4+3])>ex_threshold_v):
                #非首历元粗差剔除
                #print("去除粗差前观测列表: ", obslist)
                sat_out.append(i)
                #H,R,phase_bias,v,obslist=createKF_HRZ_new(obslist,rt_unix,X,ion_param,phase_bias)
                #print(si_PRN,'验前残差检验不通过',v[i*4],v[i*4+1],v[i*4+2],v[i*4+3])
                #print("去除粗差后观测列表: ",obslist)
        #验后方差校验
        if(post==True):
            out_v=[]
            if abs(v[i*4])>exthreshold_v_sigma*sqrt(R[i*4][i*4]): 
                #print(si_PRN," 验后方差校验不通过",v[i*4],4*sqrt(R[i*4][i*4]))
                out_v.append(v[i*4])
            if abs(v[i*4+1])>exthreshold_v_sigma*sqrt(R[i*4+1][i*4+1]):
                #print(si_PRN," 验后方差校验不通过",v[i*4+1],4*sqrt(R[i*4+1][i*4+1]))
                out_v.append(v[i*4+1])
            if abs(v[i*4+2])>exthreshold_v_sigma*sqrt(R[i*4+2][i*4+2]): 
                #print(si_PRN," 验后方差校验不通过",v[i*4],v[i*4+2],4*sqrt(R[i*4+2][i*4+2]))
                out_v.append(v[i*4+2])
            if abs(v[i*4+3])>exthreshold_v_sigma*sqrt(R[i*4+3][i*4+3]):
                #print(si_PRN," 验后方差校验不通过",v[i*4+3],4*sqrt(R[i*4+3][i*4+3]))
                out_v.append(v[i*4+3])
            if(len(out_v)):
                out_v.append(i)
                sat_out_post.append(out_v)

    #循环结束, 处理验前粗差
    obslist_new=obslist.copy()
    for s in sat_out:
        obslist_new.remove(obslist[s])
    #处理验后残差
    if(post==True):
        #全部校验通过
        if(len(sat_out_post)==0):
            return "KF fixed", obslist_new, t_phase_bias, v
        
        #找到最大残差值
        vmax=0.0
        v_out=0
        for s in sat_out_post:
            for v_i in range(0,len(s)-1):
                if(abs(s[v_i])>vmax):
                    v_out=s[-1]
                    vmax=s[v_i]
        #print("验后残差排除", obslist[v_out]['PRN'])
        obslist_new.remove(obslist[v_out])
        return "KF fixing", obslist_new, phase_bias,v

    return X,X_time,H,R,t_phase_bias,v,obslist_new

def createKF_XkPkQk_M(obslist,X,Pk,Qk):
    #系统模型构建
    #输入: 观测字典列表, 全局状态X, 全局方差Pk, 全局过程噪声Qk
    #输出: 滤波状态t_Xk, 滤波方差t_Pk, 滤波过程噪声t_Qk
    sat_num=len(obslist)#本历元有效观测卫星数量
    sys_sat_sum=round((X.shape[0]-7)/3)#全局状态卫星数量
    
    #本历元更新状态所用系统临时变量(依据: 在观测列表内的卫星数量)
    t_Xk=np.zeros((3*sat_num+7,1),dtype=np.float64)
    t_Pk=np.zeros((3*sat_num+7,3*sat_num+7),dtype=np.float64)
    t_Qk=np.zeros((3*sat_num+7,3*sat_num+7),dtype=np.float64)
    
    #有效卫星索引(计算本历元有效卫星各状态量在总状态中的索引)
    sat_use=[]#首先保证不变状态量保存在内
    for i in range(len(obslist)):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        sat_use.append([si_PRN[0],PRN_index])#系统标识符和PRN位置索引
    index_use=[0,1,2,3,4]#基础导航状态(X Y Z Dt ZWD)
    sys_shift=0
    for s in sat_use:
        if(s[0]=="C"):
            sys_shift=3*32
        if(s[0]=="E"):
            sys_shift=3*32+3*65
        index_use.append(5+sys_shift+s[1])               #电离层状态导入
    for s in sat_use:
        if(s[0]=="G"):
            sys_shift=32
        if(s[0]=="C"):
            sys_shift=3*32+65
        if(s[0]=="E"):
            sys_shift=3*32+3*65+37
        index_use.append(5+sys_shift+s[1])   #L1模糊度状态导入
    for s in sat_use:
        if(s[0]=="G"):
            sys_shift=2*32
        if(s[0]=="C"):
            sys_shift=3*32+2*65
        if(s[0]=="E"):
            sys_shift=3*32+3*65+2*37
        index_use.append(5+sys_shift+s[1]) #L2模糊度状态导入
    #ISB导入
    index_use.append(5+3*sys_sat_sum)
    index_use.append(5+3*sys_sat_sum+1)
    
    #系统状态,方差,过程噪声赋值
    for i in range(5+3*sat_num+2):
        t_Xk[i]=X[index_use[i]]                         #系统状态
        t_Qk[i][i]=Qk[index_use[i]][index_use[i]]       #系统过程噪声
        for j in range(5+3*sat_num+2):
            t_Pk[i][j]=Pk[index_use[i]][index_use[j]]   #系统方差
    #ISB赋值
    
    #返回系统模型各临时矩阵
    return t_Xk,t_Pk,t_Qk

def upstateKF_XkPkQk_M(obslist,rt_unix,t_Xk,t_Pk,t_Qk,X,Pk,Qk,X_time):
    #系统模型恢复与更新
    #输入: 观测字典列表, 滤波状态t_Xk, 滤波方差t_Pk, 滤波过程噪声t_Qk, 全局状态X, 全局方差Pk, 全局过程噪声Qk
    #输出: 恢复并更新后的全局状态X, 全局方差Pk, 全局过程噪声Qk
    sat_num=len(obslist)#本历元有效观测卫星数量
    sys_sat_sum=round((X.shape[0]-7)/3)#全局状态卫星数量
    
    #本历元更新状态所用系统临时变量(不能占用全局状态储存空间)
    t_X=np.zeros((3*sys_sat_sum+7,1),dtype=np.float64)
    t_X_time=np.zeros(3*sys_sat_sum+7,dtype=np.float64)
    t_P=np.zeros((3*sys_sat_sum+7,3*sys_sat_sum+7),dtype=np.float64)
    t_Q=np.zeros((3*sys_sat_sum+7,3*sys_sat_sum+7),dtype=np.float64)
    #拷贝原值
    t_X=X.copy()
    t_X_time=X_time.copy()
    t_P=Pk.copy()
    t_Q=Qk.copy()

    #有效卫星索引(计算本历元有效卫星各状态量在总状态中的索引)
    sat_use=[]#首先保证不变状态量保存在内
    for i in range(len(obslist)):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        sat_use.append([si_PRN[0],PRN_index])
    
    index_use=[0,1,2,3,4]
    sys_shift=0
    for s in sat_use:
        if(s[0]=="C"):
            sys_shift=3*32
        if(s[0]=="E"):
            sys_shift=3*32+3*65
        index_use.append(5+sys_shift+s[1])               #电离层状态导入
    for s in sat_use:
        if(s[0]=="G"):
            sys_shift=32
        if(s[0]=="C"):
            sys_shift=3*32+65
        if(s[0]=="E"):
            sys_shift=3*32+3*65+37
        index_use.append(5+sys_shift+s[1])   #L1模糊度状态导入
    for s in sat_use:
        if(s[0]=="G"):
            sys_shift=2*32
        if(s[0]=="C"):
            sys_shift=3*32+2*65
        if(s[0]=="E"):
            sys_shift=3*32+3*65+2*37
        index_use.append(5+sys_shift+s[1]) #L2模糊度状态导入
    #ISB导入
    index_use.append(5+3*sys_sat_sum)
    index_use.append(5+3*sys_sat_sum+1)
    
    #系统状态,方差,过程噪声恢复更新
    for i in range(5+3*sat_num+2):
        t_X[index_use[i]]=t_Xk[i]                         #系统状态
        t_X_time[index_use[i]]=rt_unix                    #系统状态时标
        t_Q[index_use[i]][index_use[i]]=t_Qk[i][i]        #系统过程噪声
        for j in range(5+3*sat_num+2):
            #滤波方差异常值处理
            if(i==j and t_Pk[i][j]<0.0):
                t_Pk[i][j]=60*60
            t_P[index_use[i]][index_use[j]]=t_Pk[i][j]   #系统方差

    #返回系统模型各全局矩阵
    return t_X,t_P,t_Q,t_X_time

#多系统PPP滤波器构建
def KF_UCPPP_M(X,X_time,Pk,Qk,ion_param,peph_sat_pos,rnx_obs,ex_threshold_v,ex_threshold_sigma,rt_unix,phase_bias,freqs):
    #扩展卡尔曼滤波
    for KF_times in range(100):
        #相位改正值拷贝
        t_phase_bias=phase_bias.copy()
        
        #观测模型(两次构建, 验前粗差剔除)
        #print(rnx_obs)
        X,X_time,H,R,_,v,rnx_obs=createKF_HRZ_M(rnx_obs,rt_unix,X,X_time,Pk,Qk,ion_param,t_phase_bias,peph_sat_pos,freqs=freqs,exthreshold_v_sigma=ex_threshold_sigma,post=False,ex_threshold_v=ex_threshold_v)
        if(not len(rnx_obs)):
            #无先验通过数据
            #全部状态重置
            X[0]=100.0
            X[1]=100.0
            X[2]=100.0
            Pk[0][0]=3600
            Pk[1][1]=3600
            Pk[2][2]=3600
            Pk[3][3]=3600
            for i in range(len(X)):
                X_time[i]=0.0
                break
        X,X_time,H,R,_,v,rnx_obs=createKF_HRZ_M(rnx_obs,rt_unix,X,X_time,Pk,Qk,ion_param,t_phase_bias,peph_sat_pos,freqs=freqs,exthreshold_v_sigma=ex_threshold_sigma,post=False,ex_threshold_v=ex_threshold_v)
        
        #系统模型(根据先验抗差得到的数据)
        t_Xk,t_Pk,t_Qk=createKF_XkPkQk_M(rnx_obs,X,Pk,Qk)

        #抗差滤波准备
        #R=IGGIII(v,R)
        #扩展卡尔曼滤波
        #1.状态一步预测
        PHIk_1_k=np.eye(t_Xk.shape[0],dtype=np.float64)
        X_up=PHIk_1_k.dot(t_Xk)
        #2.状态一步预测误差
        Pk_1_k=(PHIk_1_k.dot(t_Pk)).dot(PHIk_1_k.T)+t_Qk
        #3.滤波增益计算
        Kk=(Pk_1_k.dot(H.T)).dot(inv((H.dot(Pk_1_k)).dot(H.T)+R))
        #滤波结果
        Xk_dot=X_up+Kk.dot(v)
        #滤波方差更新
        t_Pk=((np.eye(t_Xk.shape[0],dtype=np.float64)-Kk.dot(H))).dot(Pk_1_k)  
        t_Pk=t_Pk.dot((np.eye(t_Xk.shape[0],dtype=np.float64)-Kk.dot(H)).T)+Kk.dot(R).dot(Kk.T)
        #滤波暂态更新
        t_Xk_f,t_Pk_f,t_Qk_f,t_X_time=upstateKF_XkPkQk_M(rnx_obs,rt_unix,Xk_dot,t_Pk,t_Qk,X,Pk,Qk,X_time)
        
        #验后方差
        info='KF fixed'
        info,rnx_obs,tt_phase_bias,v=createKF_HRZ_M(rnx_obs,rt_unix,t_Xk_f,t_X_time,t_Pk_f,t_Qk_f,ion_param,t_phase_bias,peph_sat_pos,freqs=freqs,exthreshold_v_sigma=ex_threshold_sigma,post=True)
        #_,info=get_post_v(rnx_obs,rt_unix,Xk_dot,ion_param,phase_bias)
        if(info=='KF fixed'):    
            #验后校验通过
            X,Pk,Qk,X_time=upstateKF_XkPkQk_M(rnx_obs,rt_unix,Xk_dot,t_Pk,t_Qk,X,Pk,Qk,X_time)
            phase_bias=tt_phase_bias.copy()
            break

    return X,Pk,Qk,X_time,v,phase_bias,rnx_obs

#多系统基于精密星历的单点定位
def SPP_from_IGS_M(obs_mats,obs_index,IGS,CLK,sat_out,ion_param,sat_pcos,freqs,sol_mode='SF',el_threthod=7.0,obslist=[],pre_rr=[]):
    rr=[100,100,100]
    #观测值列表构建(异常值剔除选星)
    if(not len(obslist)):
        obslist=[]
        #从各分系统中读取数据
        for obs_mat in obs_mats:
            for i in range(len(obs_mat[obs_index][1])):
                obsdata=obs_mat[obs_index][1][i]['OBS']
                obshealth=1
                if(obsdata[0]==0.0 or obsdata[1]==0.0 or obsdata[5]==0.0 or obsdata[6]==0.0):
                    obshealth=0
                if(obshealth):
                    if obs_mat[obs_index][1][i]['PRN'] not in sat_out:
                        obslist.append(obs_mat[obs_index][1][i])
    
    obslist_new=obslist.copy()#高度角截至列表
    sat_num=len(obslist)
    sat_prns=[t['PRN'] for t in obslist]
    sat_num_G=0
    sat_num_C=0
    sat_num_E=0
    for p in sat_prns:
        if "G" in p:
            sat_num_G=sat_num_G+1
        if "C" in p:
            sat_num_C=sat_num_C+1
        if "E" in p:
            sat_num_E=sat_num_E+1
    
    ex_index=np.zeros(sat_num,dtype=int)
    
    #方程满秩校验(三系统观测模型要求单个系统大于等于4, 且其余系统至少观测到1颗卫星)
    if(sat_num_G<4):
        print("The number of GPS < 4, pass epoch.")
        return [0,0,0,0],[],[]
    if(sat_num_C<1):
        pass
        #print("The number of BDS < 1, ISB_BDS no observations, set ISB_BDS=0.")
        #return [0,0,0,0],[],[]
    if(sat_num_E<1):
        pass
        #print("The number of GAL < 1, ISB_GAL no observations, set ISB_GAL=0.")
        #return [0,0,0,0],[],[]
    
    #卫星列表构建
    peph_sat_pos={}
    for i in range(0,sat_num):
        #光速
        clight=2.99792458e8
        #观测时间&观测值
        rt_week=obs_mats[0][obs_index][0]['GPSweek']
        rt_sec=obs_mats[0][obs_index][0]['GPSsec']
        rt_unix=satpos.gpst2time(rt_week,rt_sec)
        
        #计算卫星速度的间隔时间
        dt=0.001

        #计算精密星历历元间隔
        IGS_interval=IGS[1]['GPSsec']-IGS[0]['GPSsec']
        if(IGS_interval<0):
            IGS_interval=IGS[2]['GPSsec']-IGS[1]['GPSsec']
        
        #计算精密钟差历元间隔
        CLK_interval=CLK[1]['GPSsec']-CLK[0]['GPSsec']
        if(CLK_interval<0):
            CLK_interval=CLK[2]['GPSsec']-CLK[1]['GPSsec']
        IGS_interval=round(IGS_interval)
        CLK_interval=round(CLK_interval)
        
        #原始伪距
        p1=obslist[i]['OBS'][0]
        s1=obslist[i]['OBS'][4]
        p2=obslist[i]['OBS'][5]
        s2=obslist[i]['OBS'][9]
        #print(p1,p2,phi1,phi2)
            
        #卫星位置内插
        si_PRN=obslist[i]['PRN']#此处为卫星PRN
        #频率分发
        if("G" in si_PRN):
            f1=freqs[0][0]
            f2=freqs[0][1]
        if("C" in si_PRN):
            f1=freqs[1][0]
            f2=freqs[1][1]
        if("E" in si_PRN):
            f1=freqs[2][0]
            f2=freqs[2][1] 
        
        rs1=insert_satpos_froom_sp3(IGS,rt_unix-p1/clight,si_PRN,sp3_interval=IGS_interval)    #观测历元卫星位置
        rs2=insert_satpos_froom_sp3(IGS,rt_unix-p1/clight+dt,si_PRN,sp3_interval=IGS_interval) #插值求解卫星速度矢量
        rs=[rs1[si_PRN][0],rs1[si_PRN][1],rs1[si_PRN][2]]
        dts=insert_clk_from_sp3(CLK,rt_unix-p1/clight,si_PRN,CLK_interval)[si_PRN]
        drs=[(rs2[si_PRN][0]-rs[0])/dt,(rs2[si_PRN][1]-rs[1])/dt,(rs2[si_PRN][2]-rs[2])/dt]
        dF=-2/clight/clight*( rs[0]*drs[0]+rs[1]*drs[1]+rs[2]*drs[2] )      #利用精密星历进行相对论效应改正
        dts=dts+dF

        rs1=insert_satpos_froom_sp3(IGS,rt_unix-p1/clight-dts,si_PRN,sp3_interval=IGS_interval)    #观测历元卫星位置
        rs2=insert_satpos_froom_sp3(IGS,rt_unix-p1/clight-dts+dt,si_PRN,sp3_interval=IGS_interval) #插值求解卫星速度矢量
        rs=[rs1[si_PRN][0],rs1[si_PRN][1],rs1[si_PRN][2]]
        dts=insert_clk_from_sp3(CLK,rt_unix-p1/clight-dts,si_PRN,CLK_interval)[si_PRN]
        drts=insert_clk_from_sp3(CLK,rt_unix-p1/clight-dts,si_PRN,CLK_interval)[si_PRN]
        drs=[(rs2[si_PRN][0]-rs[0])/dt,(rs2[si_PRN][1]-rs[1])/dt,(rs2[si_PRN][2]-rs[2])/dt]
        dF=-2/clight/clight*( rs[0]*drs[0]+rs[1]*drs[1]+rs[2]*drs[2] )      #利用精密星历进行相对论效应改正
        dts=dts+dF

        # #太阳位置
        rsun,_,_=sun_moon_pos(rt_unix-p1/clight-dts+gpst2utc(rt_unix-p1/clight-dts))

        #/* unit vectors of satellite fixed coordinates */
        r=np.array([-rs[0],-rs[1],-rs[2]])
        ez=r/np.linalg.norm(r)
        r=np.array([rsun[0]-rs[0],rsun[1]-rs[1],rsun[2]-rs[2]])
        es=r/np.linalg.norm(r)
        r=np.cross(ez,es)
        ey=r/np.linalg.norm(r)
        ex=np.cross(ey,ez)

        gamma=f1*f1/f2/f2
        C1=gamma/(gamma-1.0)
        C2=-1.0 /(gamma-1.0)

        #选择卫星PCO参数
        if("G" in si_PRN):
            obs_mat=obs_mats[0]
        if("C" in si_PRN):
            obs_mat=obs_mats[1]
        if("E" in si_PRN):
            obs_mat=obs_mats[2]
        PCO_F1='L'+obs_mat[obs_index][0]['obstype'][0][1]
        PCO_F2='L'+obs_mat[obs_index][0]['obstype'][5][1]
        pco_params=sat_pcos[si_PRN]
        for param in pco_params:
            if(rt_unix-p1/clight-dts> param['Stime']):
                try:
                    off1=param[PCO_F1]
                    off2=param[PCO_F2]
                except:
                    off1=[0.0,0.0,0.0]
                    off2=[0.0,0.0,0.0]
        dant=[0.0,0.0,0.0]
        for k in range(3):
            dant1=off1[0]*ex[k]+off1[1]*ey[k]+off1[2]*ez[k]
            dant2=off2[0]*ex[k]+off2[1]*ey[k]+off2[2]*ez[k]
            dant[k]=C1*dant1+C2*dant2
        rs[0]=rs[0]+dant[0]
        rs[1]=rs[1]+dant[1]
        rs[2]=rs[2]+dant[2]
        peph_sat_pos[si_PRN]=[rs[0],rs[1],rs[2],dts,drs[0],drs[1],drs[2],(drts-dts)/dt]
    
    if(sol_mode=="Sat only"):
        return peph_sat_pos
        
    
    #伪距单点定位
    if(len(pre_rr)):
        #有先验位置
        rr[0]=pre_rr[0]
        rr[1]=pre_rr[1]
        rr[2]=pre_rr[2]
    result=np.zeros((4+2),dtype=np.float64) #结果维数4+2(X Y Z Dt ISB_BDS ISB_GAL)
    result[0:3]=rr
    result[3]=1.0   #GPS_Dt
    result[4]=0.0   #ISB_BDS
    result[5]=0.0   #ISB_GAL
    if(len(pre_rr)):
        result[3]=pre_rr[3]
    
    #print("标准单点定位求解滤波状态初值")
    #最小二乘求解滤波初值
    ls_count=0
    while(1):
        #光速, GPS系统维持的地球自转角速度(弧度制)
        clight=2.99792458e8
        OMGE=7.2921151467E-5

        #观测值矩阵初始化
        Z=np.zeros(sat_num,dtype=np.float64)
        #设计矩阵初始化
        H=np.zeros((sat_num,4+2),dtype=np.float64)
        #单位权中误差矩阵初始化
        var=np.zeros((sat_num,sat_num),dtype=np.float64)
        #权重矩阵初始化
        W=np.zeros((sat_num,sat_num),dtype=np.float64)
    
        #观测值、设计矩阵构建
        for i in range(0,sat_num):
        
            #观测时间&观测值
            rt_week=obs_mat[obs_index][0]['GPSweek']
            rt_sec=obs_mat[obs_index][0]['GPSsec']
            rt_unix=satpos.gpst2time(rt_week,rt_sec)
            #print(rt_week,rt_sec,rt_unix)
        
            #伪距
            p1=obslist[i]['OBS'][0]
            s1=obslist[i]['OBS'][4]
            p2=obslist[i]['OBS'][5]
            s2=obslist[i]['OBS'][6]
            #print(p1,p2,phi1,phi2)
            
            #卫星位置
            si_PRN=obslist[i]['PRN']
            rs=[peph_sat_pos[si_PRN][0],peph_sat_pos[si_PRN][1],peph_sat_pos[si_PRN][2]]
            dts=peph_sat_pos[si_PRN][3]
            
            r0=sqrt( (rs[0]-rr[0])*(rs[0]-rr[0])+(rs[1]-rr[1])*(rs[1]-rr[1])+(rs[2]-rr[2])*(rs[2]-rr[2]) )
            #线性化的站星单位向量
            urs_x=(rr[0]-rs[0])/r0
            urs_y=(rr[1]-rs[1])/r0
            urs_z=(rr[2]-rs[2])/r0
            
            #单卫星设计矩阵赋值与ISB赋值
            isb=0.0
            if("G" in si_PRN):
                H[i]=[urs_x,urs_y,urs_z,1.0,0.0,0.0]
                f1=freqs[0][0]
                f2=freqs[0][1]
            if("C" in si_PRN):
                H[i]=[urs_x,urs_y,urs_z,1.0,1.0,0.0]
                isb=result[4]
                f1=freqs[1][0]
                f2=freqs[1][1]
            if("E" in si_PRN):
                H[i]=[urs_x,urs_y,urs_z,1.0,0.0,1.0]
                isb=result[5]
                f1=freqs[2][0]
                f2=freqs[2][1]
            
            #地球自转改正到卫地距上
            r0=r0+OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/clight
            
            #观测矩阵
            if(sol_mode=='SF'):
                Z[i]=p1-r0-result[3]-isb-satpos.get_Tropdelay(rr,rs)-satpos.get_ion_GPS(rt_unix,rr,rs,ion_param)+clight*dts
            
            #双频无电离层延迟组合
            elif(sol_mode=='IF'):
                f12=f1*f1
                f22=f2*f2
                p_IF=f12/(f12-f22)*p1-f22/(f12-f22)*p2
                Z[i]=p_IF-r0-result[3]-isb-satpos.get_Tropdelay(rr,rs)+clight*dts

            #随机模型
            #var[i][i]= 0.00224*10**(-s1 / 10) 
            _,el=satpos.getazel(rs,rr)
            var[i][i]=0.3*0.3+0.3*0.3/sin(el)/sin(el)
            if(el*180.0/satpos.pi<el_threthod):
                var[i][i]=var[i][i]*100#低高度角拒止
                ex_index[i]=1
            if(ex_index[i]==1 and el*180.0/satpos.pi>=el_threthod):
                ex_index[i]=0
            
            if(sol_mode=='IF'):
                var[i][i]=var[i][i]*9
            W[i][i]=1.0/var[i][i]
        
        #最小二乘求解:
        if(sat_num_C<1 and sat_num_E>0):
            H_sub_ISBBDS=np.zeros((1,4+2),dtype=np.float64)
            var_sub_ISBBDS=np.zeros((1,sat_num+1),dtype=np.float64)
            H_sub_ISBBDS[0][4]=1.0      #ISB_BDS秩亏系数
            Z_sub_ISBBDS=0.0            #ISB_BDS秩亏观测
            var_sub_ISBBDS[0][-1]=1/0.01         #ISB_BDS秩亏方差
            H=np.concatenate((H,H_sub_ISBBDS)) #设计矩阵处理
            Z=np.append(Z,Z_sub_ISBBDS)        #观测矩阵处理
            W = np.hstack((W, np.zeros((sat_num,1)))) #方差阵增加一列
            W=np.concatenate((W,var_sub_ISBBDS))
        if(sat_num_E<1 and sat_num_C>0):
            H_sub_ISBGAL=np.zeros((1,4+2),dtype=np.float64)
            var_sub_ISBGAL=np.zeros((1,sat_num+1),dtype=np.float64)
            H_sub_ISBGAL[0][5]=1.0      #ISB_GAL秩亏系数
            Z_sub_ISBGAL=0.0            #ISB_GAL秩亏观测
            var_sub_ISBGAL[0][-1]=1/0.01         #ISB_GAL秩亏方差
            H=np.concatenate((H,H_sub_ISBGAL)) #设计矩阵处理
            Z=np.append(Z,Z_sub_ISBGAL)        #观测矩阵处理
            W = np.hstack((W, np.zeros((sat_num,1)))) #方差阵增加一列
            W=np.concatenate((W,var_sub_ISBGAL))
        if(sat_num_E<1 and sat_num_C<1):
            H_sub_ISBBDS=np.zeros((1,4+2),dtype=np.float64)
            var_sub_ISBBDS=np.zeros((1,sat_num+1),dtype=np.float64)
            H_sub_ISBBDS[0][4]=1.0      #ISB_BDS秩亏系数
            Z_sub_ISBBDS=0.0            #ISB_BDS秩亏观测
            var_sub_ISBBDS[0][-1]=1.0/0.01         #ISB_BDS秩亏方差
            H=np.concatenate((H,H_sub_ISBBDS)) #设计矩阵处理
            Z=np.append(Z,Z_sub_ISBBDS)        #观测矩阵处理
            W = np.hstack((W, np.zeros((sat_num,1))))
            W=np.concatenate((W,var_sub_ISBBDS))    #方差阵增加一列
            
            H_sub_ISBGAL=np.zeros((1,4+2),dtype=np.float64)
            var_sub_ISBGAL=np.zeros((1,sat_num+2),dtype=np.float64)
            H_sub_ISBGAL[0][5]=1.0      #ISB_GAL秩亏系数
            Z_sub_ISBGAL=0.0            #ISB_GAL秩亏观测
            var_sub_ISBGAL[0][-1]=1/0.01         #ISB_GAL秩亏方差
            H=np.concatenate((H,H_sub_ISBGAL)) #设计矩阵处理
            Z=np.append(Z,Z_sub_ISBGAL)        #观测矩阵处理
            W = np.hstack((W, np.zeros((sat_num+1,1)))) #方差阵增加一列
            W=np.concatenate((W,var_sub_ISBGAL))      #方差阵增加一行
        
        dresult=getLSQ_solution(H,Z,W=W,weighting_mode='S')
        
        #迭代值更新
        result[0]+=dresult[0]
        result[1]+=dresult[1]
        result[2]+=dresult[2]
        result[3]+=dresult[3] #GPS_CLK
        result[4]+=dresult[4] #ISB_BDS
        result[5]+=dresult[5] #ISB_GAL

        #更新测站位置
        rr[0]=result[0]
        rr[1]=result[1]
        rr[2]=result[2]
        #print(dresult)
        ls_count+=1
        if(abs(dresult[0])<1e-4 and abs(dresult[1])<1e-4 and abs(dresult[2])<1e-4):
            #估计先验精度因子
            break
        
        if(ls_count>200):
            #最小二乘迭代次数
            break
    
    #排除低高度角卫星
    for i in range(sat_num):
        if(ex_index[i]):
            obslist_new.remove(obslist[i])
    return result,obslist_new,peph_sat_pos

def update_phase_slip_M(obslist,GF_sign,Mw_sign,slip_sign,Mw_threshold,GF_threshold,freqs,dN=[],dN_fix_mode=0):
    #首先清空周跳标志
    for i in range(len(slip_sign)):
        slip_sign[i]=0
    
    #清空无观测值的周跳检测量
    prns=[t['PRN'] for t in obslist]    #字符型PRN列表
    for i in range(len(GF_sign)):       #遍历各系统各卫星周跳标志
        if(i<32):
            in_PRN="G{:02d}".format(i+1)
        elif(i<32+65):
            in_PRN="C{:02d}".format(i-32+1)
        elif(i<32+65+37):
            in_PRN="E{:02d}".format(i-32-65+1)
        if(in_PRN not in prns):
            GF_sign[i]=0.0
            Mw_sign[i]=0.0
    
    #周跳检测
    sat_num=len(obslist)
    
    for i in range(sat_num):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        f_ids=0
        if("C" in si_PRN):
            PRN_index=PRN_index+32
            f_ids=1
        if("E" in si_PRN):
            PRN_index=PRN_index+32+65
            f_ids=2
        
        p1=obslist[i]['OBS'][0]
        l1=obslist[i]['OBS'][1]
        p2=obslist[i]['OBS'][5]
        l2=obslist[i]['OBS'][6]

        GF,Mw,slip,dN1,dN2=get_phase_jump(p1,p2,l1,l2,GF_sign[PRN_index],Mw_sign[PRN_index],Mw_threshold,GF_threshold,f1=freqs[f_ids][0],f2=freqs[f_ids][1])
        if(slip):
            #print('{} 发生周跳 GF:{}->{} Mw:{}->{} p1:{} l1:{} p2:{} l2:{} dN1:{} dN2:{}'.format(si_PRN,GF_sign[PRN_index],GF,Mw_sign[PRN_index],Mw,p1,l1,p2,l2,dN1,dN2))
            pass
            #print('{} phase jump occurred, GF:{}->{} Mw:{}->{} p1:{} l1:{} p2:{} l2:{} dN1:{} dN2:{}'.format(si_PRN,GF_sign[PRN_index],GF,Mw_sign[PRN_index],Mw,p1,l1,p2,l2,dN1,dN2))
        GF_sign[PRN_index]=GF
        Mw_sign[PRN_index]=Mw
        slip_sign[PRN_index]=slip
        
        if(dN_fix_mode):
            dN[PRN_index][0]=dN1
            dN[PRN_index][1]=dN2
    
    return GF_sign,Mw_sign,slip_sign,dN

def updata_PPP_state_M(X,Pk,spp_rr,epoch,rt_unix,X_time,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,GF_threshold,Mw_threshold,
                     sat_num,rnx_obs,out_age,freqs,dy_mode):
    #状态递推
    X[3]=spp_rr[3]#接收机钟差数值更新
    X[-2]=spp_rr[4]#系统间偏差数值更新(ISB_BDS)
    X[-1]=spp_rr[5]#系统间偏差数值更新(ISB_GAL)
    
    if(X[0]==100.0):
        #上历元先验重置
        print("上历元无解,重置")
        X[0]=spp_rr[0]
        X[1]=spp_rr[1]
        X[2]=spp_rr[2]
    
    
    if(dy_mode!='static'):
        #非静态观测
        X[0]=spp_rr[0]    
        X[1]=spp_rr[1]    
        X[2]=spp_rr[2]    
    
    #非首历元, 状态重置
    if(epoch):
        #计算位置/钟差/对流层状态更新时间差
        dt=rt_unix-X_time[0][0]
        #位置/钟差/对流层状态过程噪声
        if(dy_mode=='static'):
            Qk[0][0]=1e-8#3600.0#坐标改正数
            Qk[1][1]=1e-8#3600.0#坐标改正数
            Qk[2][2]=1e-8#3600.0#坐标改正数
            Qk[3][3]=60*60#接收机钟差(白噪声)
            Qk[4][4]=1e-8*dt#对流层延迟(缓慢变化)
            Qk[-2][-2]=60*60#接收机钟差(ISB_BDS, 白噪声)
            Qk[-1][-1]=60*60#接收机钟差(ISB_GAL, 白噪声)
        else:
            Qk[0][0]=3600.0#坐标改正数
            Qk[1][1]=3600.0#坐标改正数
            Qk[2][2]=3600.0#坐标改正数
            Qk[3][3]=60*60#接收机钟差(白噪声)
            Qk[4][4]=1e-8*dt#对流层延迟(缓慢变化)
            Qk[-2][-2]=60*60#接收机钟差(ISB_BDS, 白噪声)
            Qk[-1][-1]=60*60#接收机钟差(ISB_GAL, 白噪声)

        #部分更新的状态量
        #GPS电离层状态范围[5:5+32]
        for j in range(5,5+32):
            dt=rt_unix-X_time[j][0]
            si_PRN="G{:02d}".format(j-5+1)                  #待更新状态的卫星PRN
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]   #观测列表中的PRNs
            if(si_PRN in rnx_obs_prns and dt<=out_age ): 
                Qk[j][j]=0.0016*dt  #电离层()
            if(si_PRN in rnx_obs_prns and dt>out_age):
                if(dt>out_age):
                    GF_sign[int(si_PRN[1:])-1]=0.0
                    Mw_sign[int(si_PRN[1:])-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                ion=update_ion(p1,p2,f1=freqs[0][0],f2=freqs[0][1])
                X[j]=ion       #重置垂直电离层估计
                Qk[j][j]=60*60                    #重置过程噪声
                Pk[j][j]=60*60
        #BDS电离层状态范围[5+3*32,5+3*32+65]
        for j in range(5+3*32,5+3*32+65):
            dt=rt_unix-X_time[j][0]
            si_PRN="C{:02d}".format(j-5-3*32+1)#整型PRN序号(PRN-1)
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age ): 
                Qk[j][j]=0.0016*dt  #电离层()
            if(si_PRN in rnx_obs_prns and dt>out_age):
                if(dt>out_age):
                    GF_sign[32+int(si_PRN[1:])-1]=0.0
                    Mw_sign[32+int(si_PRN[1:])-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                ion=update_ion(p1,p2,f1=freqs[1][0],f2=freqs[1][1])
                X[j]=ion       #重置垂直电离层估计
                Qk[j][j]=60*60                    #重置过程噪声
                Pk[j][j]=60*60
        #GAL电离层状态范围[5+3*32+3*65,5+3*32+3*65+37]
        for j in range(5+3*32+3*65,5+3*32+3*65+37):
            dt=rt_unix-X_time[j][0]
            si_PRN="E{:02d}".format(j-5-3*32-3*65+1)#整型PRN序号(PRN-1)
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age ): 
                Qk[j][j]=0.0016*dt  #电离层()
            if(si_PRN in rnx_obs_prns and dt>out_age):
                if(dt>out_age):
                    GF_sign[32+65+int(si_PRN[1:])-1]=0.0
                    Mw_sign[32+65+int(si_PRN[1:])-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                ion=update_ion(p1,p2,f1=freqs[2][0],f2=freqs[2][1])
                X[j]=ion       #重置垂直电离层估计
                Qk[j][j]=60*60                    #重置过程噪声
                Pk[j][j]=60*60
        
        #GPS 第一频率模糊度[5+32,5+2*32]
        for j in range(5+32,5+2*32):
            dt=rt_unix-X_time[j][0]
            si_PRN="G{:02d}".format(j-(5+32)+1)              #PRN码
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=1e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p1,l1,freqs[0][0],p1,p2,f1=freqs[0][0],f2=freqs[0][1])
                Qk[j][j]=60*60   #重置过程噪声
                Pk[j][j]=60*60
        #BDS 第一频率模糊度[5+3*32+65,5+3*32+2*65]
        for j in range(5+3*32+65,5+3*32+2*65):
            dt=rt_unix-X_time[j][0]
            si_PRN="C{:02d}".format(j-(5+3*32+65)+1)              #PRN码
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=1e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p1,l1,freqs[1][0],p1,p2,f1=freqs[1][0],f2=freqs[1][1])
                Qk[j][j]=60*60   #重置过程噪声
                Pk[j][j]=60*60
        #GAL 第一频率模糊度[5+3*32+3*65+37,5+3*32+3*65+2*37]
        for j in range(5+3*32+3*65+37,5+3*32+3*65+2*37):
            dt=rt_unix-X_time[j][0]
            si_PRN="E{:02d}".format(j-(5+3*32+3*65+37)+1)              #PRN码
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=1e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p1,l1,freqs[2][0],p1,p2,f1=freqs[2][0],f2=freqs[2][1])
                Qk[j][j]=60*60   #重置过程噪声
                Pk[j][j]=60*60                
        
        #GPS 第二频率模糊度[5+2*32,5+3*32]
        for j in range(5+2*32,5+3*32):
            dt=rt_unix-X_time[j][0]
            si_PRN="G{:02d}".format(j-(5+2*32)+1)   #PRN
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的PRN
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=1e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p2,l2,freqs[0][1],p1,p2,f1=freqs[0][0],f2=freqs[0][1])
                Qk[j][j]=60*60   #重置过程噪声
                Pk[j][j]=60*60
        #BDS 第二频率模糊度[5+3*32+2*65,5+3*32+3*65]
        for j in range(5+3*32+2*65,5+3*32+3*65):
            dt=rt_unix-X_time[j][0]
            si_PRN="C{:02d}".format(j-(5+3*32+2*65)+1)#整型PRN序号(PRN-1)
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=1e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p2,l2,freqs[1][1],p1,p2,f1=freqs[1][0],f2=freqs[1][1])
                Qk[j][j]=60*60   #重置过程噪声
                Pk[j][j]=60*60
        #GAL 第二频率模糊度[5+3*32+3*65+2*37,5+3*32]
        for j in range(5+3*32+3*65+2*37,5+3*32+3*65+3*37):
            dt=rt_unix-X_time[j][0]
            si_PRN="E{:02d}".format(j-(5+3*32+3*65+2*37))#整型PRN序号(PRN-1)
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=1e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p2,l2,freqs[2][1],p1,p2,f1=freqs[2][0],f2=freqs[2][1])
                Qk[j][j]=60*60   #重置过程噪声
                Pk[j][j]=60*60        
        #周跳探测
        #GF_sign,Mw_sign,slip_sign,dN_sign=
        update_phase_slip_M(rnx_obs,GF_sign,Mw_sign,slip_sign,Mw_threshold,GF_threshold,freqs,dN_sign,dN_fix_mode=1)
        #小周跳修复/大周跳重置
        for j in range(len(slip_sign)):
            # if(slip_sign[j] and (abs(dN_sign[j][0])<500 or abs(dN_sign[j][1]<500))):
            #     #print('{} G{:02d} 周跳修复 GF: {} Mw:{} dN1:{} dN2:{}'.format(epoch,j+1,GF_sign[j],Mw_sign[j],dN_sign[j][0],dN_sign[j][1]))
            #     X[5+sat_num+j]=X[5+sat_num+j]+dN_sign[j][0]*clight/f1                
            #     X[5+2*sat_num+j]=X[5+2*sat_num+j]+dN_sign[j][1]*clight/f2
            #     Qk[5+sat_num+j][5+sat_num+j]=1e2                
            #     Qk[5+2*sat_num+j][5+2*sat_num+j]=1e2
            if(slip_sign[j] and (abs(dN_sign[j][0])>=0.0 or abs(dN_sign[j][1]>=0.0))):
                sys_shift=0
                sys_in_num=32
                sys_in_id=0
                freqs_id=0
                if(j<32):
                    si_PRN="G{:02d}".format(j+1)
                    sys_in_id=j
                elif(j<32+65):
                    si_PRN="C{:02d}".format(j-32+1)
                    sys_shift=32
                    sys_in_num=65
                    sys_in_id=j-32
                    freqs_id=1
                elif(j<32+65+37):
                    si_PRN="E{:02d}".format(j-32-65+1)
                    sys_shift=32+65
                    sys_in_num=37
                    sys_in_id=j-32-65
                    freqs_id=2
                rnx_obs_prns=[t['PRN'] for t in rnx_obs]
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                #周跳状态量与过程噪声更新
                X[5+3*sys_shift+sys_in_num+sys_in_id]=update_phase_amb(p1,l1,freqs[freqs_id][0],p1,p2,f1=freqs[freqs_id][0],f2=freqs[freqs_id][1])              
                X[5+3*sys_shift+2*sys_in_num+sys_in_id]=update_phase_amb(p2,l2,freqs[freqs_id][1],p1,p2,f1=freqs[freqs_id][0],f2=freqs[freqs_id][1])
                Qk[5+3*sys_shift+sys_in_num+sys_in_id][5+3*sys_shift+sys_in_num+sys_in_id]=60*60                
                Qk[5+3*sys_shift+2*sys_in_num+sys_in_id][5+3*sys_shift+2*sys_in_num+sys_in_id]=60*60

def log2out_M(rt_unix,v,obslist,X,X_time,Pk,peph_sat_pos,freqs):
    #历元数据整备
    out={}
    for i in range(len(obslist)):
        si_PRN=obslist[i]['PRN']            #提取卫星PRN码
        PRN_index=int(si_PRN[1:])-1         #卫星PRN码对应的系统内偏置
        sys_sat_num=int((X.shape[0]-5)/3)   #总卫星数

        p1=obslist[i]['OBS'][0]
        l1=obslist[i]['OBS'][1]
        s1=obslist[i]['OBS'][4]
        p2=obslist[i]['OBS'][5]
        l2=obslist[i]['OBS'][6]
        s2=obslist[i]['OBS'][9]
        #print(p1,p2,phi1,phi2)
        
        #卫星位置
        rs=[peph_sat_pos[si_PRN][0],peph_sat_pos[si_PRN][1],peph_sat_pos[si_PRN][2],peph_sat_pos[si_PRN][3]]

        out[si_PRN]={}
        out[si_PRN]['GPSweek'],out[si_PRN]['GPSsec']=satpos.time2gpst(rt_unix)
        
        out[si_PRN]['sat_x']=rs[0]
        out[si_PRN]['sat_y']=rs[1]
        out[si_PRN]['sat_z']=rs[2]
        out[si_PRN]['sat_cdt']=satpos.clight*rs[3]

        #测站坐标
        out[si_PRN]['sta_x']=X[0][0]
        out[si_PRN]['std_sta_x']=Pk[0][0]
        out[si_PRN]['sta_y']=X[1][0]
        out[si_PRN]['std_sta_y']=Pk[1][1]
        out[si_PRN]['sta_z']=X[2][0]
        out[si_PRN]['std_sta_z']=Pk[2][2]
        
        #GPS钟差
        out[si_PRN]['GPSsec_dt']=X[3][0]
        out[si_PRN]['std_GPSsec_dt']=Pk[3][3]
        
        #BDS钟差改正数
        out[si_PRN]['ISB_BDS']=X[5+3*sys_sat_num][0]
        out[si_PRN]['std_ISB_BDS']=Pk[5+3*sys_sat_num][5+3*sys_sat_num]

        #GAL钟差改正数
        out[si_PRN]['ztd_w']=X[4][0]
        out[si_PRN]['std_ztd_w']=Pk[4][4]

        rr=[X[0][0],X[1][0],X[2][0]]
        
        #天顶对流层干延迟
        out[si_PRN]['ztd_h']=get_Trop_delay_dry(rr)
        
        #滤波后验残差
        out[si_PRN]['res_p1']=v[4*i][0]
        out[si_PRN]['res_l1']=v[4*i+1][0]
        out[si_PRN]['res_p2']=v[4*i+2][0]
        out[si_PRN]['res_l2']=v[4*i+3][0]

        #站星几何关系
        az,el=getazel(rs,rr)
        out[si_PRN]['azel']=[az/pi*180.0,el/pi*180.0]

        #电离层状态更新
        sys_shift=0
        f_id=0
        if("C" in si_PRN):
            sys_shift=3*32
            f_id=1
        if("E" in si_PRN):
            sys_shift=3*32+3*65
            f_id=2
        if(X_time[5+sys_shift+PRN_index]==rt_unix):
            Mi=IMF_ion(rr,rs,MF_mode=1,H_ion=350e3)
            out[si_PRN]['STEC']=X[5+sys_shift+PRN_index][0]*(freqs[f_id][0]/1e8)*(freqs[f_id][0]/1e8)/40.28
            out[si_PRN]['std_STEC']=Pk[5+sys_shift+PRN_index][5+sys_shift+PRN_index]*((freqs[f_id][0]/1e8)*(freqs[f_id][0]/1e8)/40.28)**2
        
        #模糊度状态更新
        sys_shift=32    #N1基础状态
        if("C" in si_PRN):
            sys_shift=3*32+65
        if("E" in si_PRN):
            sys_shift=3*32+3*65+37
        if(X_time[5+sys_shift+PRN_index]==rt_unix):
            out[si_PRN]['N1']=X[5+sys_shift+PRN_index][0]
            out[si_PRN]['std_N1']=Pk[5+sys_shift+PRN_index][5+sys_shift+PRN_index]
        sys_shift=2*32  #N2基础状态
        if("C" in si_PRN):
            sys_shift=3*32+2*65
        if("E" in si_PRN):
            sys_shift=3*32+3*65+2*37
        if(X_time[5+sys_shift+PRN_index]==rt_unix):
            out[si_PRN]['N2']=X[5+sys_shift+PRN_index][0]
            out[si_PRN]['std_N2']=Pk[5+sys_shift+PRN_index][5+sys_shift+PRN_index]
    return out

def UCPPP_M(obs_mats,obs_start,obs_epoch,IGS,clk,
          sat_out,ion_param,sat_pcos,el_threthod,ex_threshold_v,ex_threshold_v_sigma,Mw_threshold,GF_threshold,dy_mode,
          X,Pk,Qk,phase_bias,X_time,GF_sign,Mw_sign,slip_sign,dN_sign,sat_num,out_age,freqs):
    
    Out_log=[]

    obs_index=obs_start
    for epoch in tqdm(range(obs_epoch)):
    
        #观测时间
        rt_week=obs_mats[0][obs_index+epoch][0]['GPSweek']
        rt_sec=obs_mats[0][obs_index+epoch][0]['GPSsec']
        rt_unix=satpos.gpst2time(rt_week,rt_sec)
    
        #进行单点定位
        spp_rr,rnx_obs,peph_sat_pos=SPP_from_IGS_M(obs_mats,obs_index+epoch,IGS,clk,sat_out,ion_param,sat_pcos,freqs,el_threthod=el_threthod,sol_mode="SF",pre_rr=[X[0][0],X[1][0],X[2][0],X[3][0]])
        #无单点定位解
        if(not len(rnx_obs)):
            print("No valid observations, Pass epoch: Week: {}, sec: {}.".format(rt_week,rt_sec))
            continue

        #PPP状态更新
        updata_PPP_state_M(X,Pk,spp_rr,epoch,rt_unix,X_time,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,GF_threshold,Mw_threshold,sat_num,rnx_obs,out_age,freqs,dy_mode)

        #PPP时间更新
        X,Pk,Qk,X_time,v,phase_bias,rnx_obs=KF_UCPPP_M(X,X_time,Pk,Qk,ion_param,peph_sat_pos,rnx_obs,ex_threshold_v,ex_threshold_v_sigma,rt_unix,phase_bias,freqs)

        #结果保存
        #Out_log.append([X[0][0],X[1][0],X[2][0]])
        Out_log.append(log2out_M(rt_unix,v,rnx_obs,X,X_time,Pk,peph_sat_pos,freqs))
    
    return Out_log


