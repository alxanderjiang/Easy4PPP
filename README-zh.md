# Easy4PPP: 纯Python编译的精密单点定位工具包
[[中]](./README-zh.md) &ensp; [[EN]](./README.md)
## 快速开始
1. 工具包提供一个示例jupyter notebook教程用于快速演示工具包进行精密单点定位（Precise Point Position, PPP）解算的结果。在使用工具包之前, 确保你已经在环境中正确安装Python发行版本和numpy库。
2. 首先将源代码中的“data.zip”压缩文件解压到同一目录下，该目录内包括一组PPP解算的示例数据。
3. 解压完成后，运行“ppp.ipynb”的全部代码块，完成后将在“nav_result”文件夹内生成一个解算结果文件（Easy4PPP提供的以numpy字典数组为数据结构的Python语言内标准数据交换格式）。
4. 生成结果文件后，运行“nav_result.ipynb”的全部代码块，在jupyter notebook中可以得到PPP收敛曲线, 站星连线斜射总电子含量散点和码残差散点图像.
## 下载与环境准备
1. **直接从github主页下载工具包的压缩文件“Easy4PPP-main.zip”并解压**即可。如果您使用git clone工具，请运行如下命令将Easy4PPP克隆到本地：
```bash
git clone https://github.com/alxanderjiang/Easy4PPP.git
```
2. Easy4PPP在工具包中提供一个示例数据文件夹压缩包：data.zip，请将该压缩包解压到Easy4PPP同路径下以正常运行示例PPP程序。如果您使用Linux且仅能通过终端操作，请运行如下命令完成解压：
 ```bash
cd Easy4PPP
unzip data.zip
```
3. 在使用Easy4PPP前，请确保您的Python环境中已经安装好numpy，tqdm和ipykernel。numpy和tqdm是使用Easy4PPP工具箱核心运算库必须的第三方库，ipykenel则是顺利运行Easy4PPP提供的示例Jupyter Notebook教程组所需。如果您的环境中尚未安装上述包，请运行如下命令安装：
 ```bash
pip install numpy
pip install tqdm
pip install ipykernel
```
如果您希望通过yaml格式的配置文件运行Easy4PPP, 还需额外在您的环境中安装Pyyaml包 (区别于下载, 在使用该包时, 应该使用import yaml), 运行如下命令: 
```bash
pip install Pyyaml
```
## 通过源代码文件使用Easy4PPP运行单GPS解
1. 对示例数据进行PPP解算可以通过在Easy4PPP工具包根目录下直接运行sppp.py或在Jupyter Notebook 中运行ppp.ipynb实现。
2. 运行结束后，nav_result文件夹内生成numpy数组格式的PPP结果日志“jfng1320.24o.out.npy”。
3. 在Jupyter Notebook中运行nav_result.ipynb即可得到可视化的PPP结果和产品。详细信息请查阅UserMannual.pdf。
## 通过配置文件运行脚本使用Easy4PPP运行多系统解
1. 对示例数据进行PPP解算还可以通过在Easy4PPP工具包根目录下使用命令行输入"python src/ppp_yaml.py xmls_mgex/Easy4PPP_JFNG_GCE.yaml" 以运行ppp_yaml.py或在 Jupyter Notebook 中运行ppp_yaml.ipynb实现。由于Github单个文件大小限制, 示例数据仅保留GPS+BDS观测。
2. 运行结果存储格式和可视化方式同上。
3. Jupyter Notebook 文件rinex2yaml提供从RINEX格式观测文件自动生成Easy4PPP的示例脚本, 该脚本会自动解析观测文件采样率以匹配最常规的PPP配置。
## 联系我们
Easy4PPP工具箱的一切内容均由武汉理工大学智能交通系统研究中心/航运学院的蒋卓君，杨泽恩，黄文静和钱闯完成。Easy4PPP目前处于测试阶段，不完善之处敬请谅解。Easy4PPP项目组欢迎一切有关工具箱的建议、意见和漏洞记录反馈，联系方式：zhuojun_jiang@whut.edu.cn或1162110359@qq.com。
