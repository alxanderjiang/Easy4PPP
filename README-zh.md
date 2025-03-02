# Easy4PPP: 纯Python编译的精密单点定位工具包
## 快速开始
1. 工具包提供一个示例jupyter notebook教程用于快速演示工具包进行精密单点定位（Precise Point Position, PPP）解算的结果。在使用工具包之前, 确保你已经在环境中正确安装Python发行版本和numpy库。
2. 首先将源代码中的“data.zip”压缩文件解压到同一目录下，该目录内包括一组PPP解算的示例数据。
3. 解压完成后，运行“ppp.ipynb”的全部代码块，完成后将在“nav_result”文件夹内生成一个解算结果文件（Easy4PPP提供的以numpy字典数组为数据结构的Python语言内标准数据交换格式）。
4. 生成结果文件后，运行“nav_result.ipynb”的全部代码块，在jupyter notebook中可以得到PPP收敛曲线, 站星连线斜射总电子含量散点和码残差散点图像.
