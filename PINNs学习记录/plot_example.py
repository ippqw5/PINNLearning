# 运行前需要 pip install matplotlib numpy 安装依赖库

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyparsing import col
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math

def set_axes3D(axes:Axes3D,title:str):
    """设置 axes 的格式

    Args:
        axes (Axes3D): _description_
        title (str): _description_
    """
    axes.set_box_aspect((3,1,3)) # 坐标轴缩放比例
    axes.set_xlim(0,1) #坐标轴范围
    axes.set_ylim(0,1)
    axes.set_zlim(0,1)
    axes.set_xlabel('$x$',fontsize = 15, labelpad=-8.0) # 设置坐标轴标签
    axes.set_ylabel('$y$',fontsize = 15, labelpad=-3.0)
    axes.set_zlabel('$z$',fontsize = 15, labelpad= 2.0)
    axes.view_init(elev=30, azim=135) # 观察视角
    axes.tick_params(
        axis = 'x',
        which = 'major',
        pad = -4.5
    ) # 特殊设置，坐标轴标签位置调整
    axes.tick_params(
        axis = 'y',
        which = 'major',
        pad = -2.0
    )
    axes.tick_params(
        axis = 'both',
        which = 'major',
        direction = 'out',
        length = 1.0,
        labelsize = 'small', #刻度标签的文字大小
    )
    # axes.xaxis.set_major_formatter('{x:.1f}s') # 还是不加单位了
    # axes.locator_params('x',nbins = 1,tight=True)
    axes.set_title(title,fontsize = 15) # 子图标题
    
def fig_plot3D_scatter(data_dict,colN = 4):
    
    figN = len(data_dict)
    rowN = math.ceil(figN /colN)
    
    subsize = [4.5,3] # 宽 高
    fig = plt.figure(figsize=(subsize[0] * colN, subsize[1] * rowN))

    for i,(key,datas) in enumerate(data_dict.items()):
        axes:Axes3D = fig.add_subplot(rowN, colN, i+1, projection='3d')
        set_axes3D(axes,key)
        img = axes.scatter(datas[0],datas[1],datas[2],
                           c = datas[3],
                           s=3,
                           cmap=plt.get_cmap('RdYlBu_r'), 
                           alpha=0.3)

        fig.colorbar(img, ax=axes, shrink=0.6, format='%.2f')
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=1.0)
    return fig

def fig_plotQuiver3D(data_dict,colN = 4):
    figN = len(data_dict)
    rowN = math.ceil(figN /colN)
    
    subsize = [4.5,3] # 宽 高
    fig = plt.figure(figsize=(subsize[0] * colN, subsize[1] * rowN))

    for i,(key,datas) in enumerate(data_dict.items()):
        axes:Axes3D = fig.add_subplot(rowN, colN, i+1, projection='3d')
        set_axes3D(axes,key)
        axes.quiver(datas[0],datas[1],datas[2], datas[3],datas[4],datas[5],length=0.1,normalize=True,linewidths=0.5)
        
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=1.0)
    return fig
    
    
def set_axes(axes,title):
    # axes.set_box_aspect(1.0)
    axes.set_xlim(0,1)
    axes.set_ylim(0,1)
    axes.set_xlabel("$x$",fontsize = 15)
    axes.set_ylabel("$y$",fontsize = 15,rotation = 'horizontal',labelpad = 8.0)
    axes.tick_params(
        axis = 'both',
        which = 'major',
        direction = 'out',
        length = 1.0,
        labelsize = 'small'
    )
    axes.set_title(title,fontsize = 15)

def fig_plot2D_pcolor(data_dict,colN = 4):
    
    figN = len(data_dict)
    rowN = math.ceil(figN /colN)
    
    subsize = [4.5,3] # 宽 高
    
    fig = plt.figure(figsize=(subsize[0] * colN, subsize[1] * rowN))
    
    for i,(key,datas) in enumerate(data_dict.items()):
        axes = fig.add_subplot(rowN, colN, i+1, projection=None)
        set_axes(axes,key)
        img = axes.pcolormesh(datas[0],datas[1],datas[2],shading='gouraud',cmap=plt.get_cmap('RdYlBu_r'), alpha=1.0)
        fig.colorbar(img, ax=axes, shrink=0.6, format='%.4g')

    plt.tight_layout()
    return fig

def fig_plot2D_scatter(data_dict,colN = 4, s = 2):
    
    figN = len(data_dict)
    rowN = math.ceil(figN /colN)
    
    subsize = [4.5,3] # 宽 高
    
    fig = plt.figure(figsize=(subsize[0] * colN, subsize[1] * rowN))
    
    for i,(key,datas) in enumerate(data_dict.items()):
        axes = fig.add_subplot(rowN, colN, i+1, projection=None)
        set_axes(axes,key)
        img = axes.scatter(datas[0],datas[1],c = datas[2],s = s,cmap=plt.get_cmap('RdYlBu_r'), alpha=1.0)
        fig.colorbar(img, ax=axes, shrink=0.6, format='%.4g')

    plt.tight_layout()
    return fig

def fig_plot2D_quiver(data_dict,colN = 4):
    
    figN = len(data_dict)
    rowN = math.ceil(figN /colN)
    
    subsize = [4.5,3] # 宽 高
    
    fig = plt.figure(figsize=(subsize[0] * colN, subsize[1] * rowN))
    
    for i,(key,datas) in enumerate(data_dict.items()):
        axes = fig.add_subplot(rowN, colN, i+1, projection=None)
        set_axes(axes,key)
        img = axes.quiver(datas[0],datas[1],datas[2],datas[3],cmap=plt.get_cmap('RdYlBu_r'), alpha=1.0)
        fig.colorbar(img, ax=axes, shrink=0.6, format='%.4g')

    plt.tight_layout()
    return fig
    
if __name__ == '__main__':
    import numpy as np
    
    X,Y,Z = np.meshgrid(np.linspace(0.0,1.0,50),np.linspace(0.0,1.0,50),np.linspace(0.0,1.0,50))
    X,Y,Z = X.flatten(),Y.flatten(),Z.flatten()
    uf_x = np.random.random(size = len(X))
    uf_y = np.random.random(size = len(X))
    uf_z = np.random.random(size = len(X))
    pf = np.random.random(size = len(X))
    
    example_3D_data_dict = {
        '$u_{f_x}$':[X,Y,Z,uf_x],
        '$u_{f_y}$':[X,Y,Z,uf_y],
        '$u_{f_z}$':[X,Y,Z,uf_z],
        '$p_f$':[X,Y,Z,pf],
    }
    
    fig = fig_plot3D_scatter(example_3D_data_dict,colN=3)
    #fig = fig_plot2D_quiver(example_3D_data_dict,colN=3)
    #plt.savefig('example_3D_data_dict_plot3D_scatter.png')

    X,Y = np.meshgrid(np.linspace(0.0,1.0,50),np.linspace(0.0,1.0,50))
    X,Y = X.flatten(),Y.flatten()
    uf_x = np.random.random(size = len(X))
    uf_y = np.random.random(size = len(X))
    pf = np.random.random(size = len(X))
    example_2D_data_dict = {
        '$u_{f_x}$':[X,Y,uf_x],
        '$u_{f_y}$':[X,Y,uf_y],
        '$p_f$':[X,Y,pf],
    }
    
    fig = fig_plot2D_scatter(example_2D_data_dict,colN=3)
    #plt.savefig('example_3D_data_dict_plot2D_scatter.png')

    plt.show()