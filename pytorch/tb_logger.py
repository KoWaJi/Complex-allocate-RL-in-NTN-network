#  tensorboard可视化
import csv  # 一种通用的电子表格和数据库导入导出格式
import fnmatch  # 该模块主要是用来做文件名称的匹配
import os  # 导入标准库os
import pprint
from io import StringIO  # 用于像文件一样对字符串缓冲区或者叫做内存文件进行读写
from threading import Lock  # 互斥锁 是可用的最低级的同步指令，lock处于锁定状态时不被特定的线程拥有，每个资源都有一个可称为互斥锁的标记，用来保证在任意时刻只有一个线程访问该资源

import matplotlib.pyplot as plt  # 常用的绘图模块
import six  # 用来解决python2和python3兼容性的问题
from tensorboard.backend.event_processing import event_accumulator   # 导入tensorboard的事件解析器
from torch.utils.tensorboard import SummaryWriter

from pytorch.util import *  # 将一个模块中的所有函数都导入进来

TMP_LOG_PATH = "tmp/"


def string_to_tb_string(markdown_string):  # markdown 轻量级标记语言？转义字符语法
    if isinstance(markdown_string, six.binary_type):
        markdown_string = markdown_string.decode('utf-8')
    markdown_string = markdown_string.replace('\n', '  \n')
    markdown_string = markdown_string.replace('\t', '    ')
    return markdown_string


def plot_cdf(mat, fig_size=(10, 8)):    # mat函数用来创建矩阵
    fig = plt.figure(figsize=fig_size)  # figuresize指定图像的宽和高
    ax = fig.add_subplot(111)           # 使用默认值创建一个子图，三位数分别表示行数、列数和第几个格子
    n = mat.shape[0] // 10              # 地板除运算，先做除法然后向下取整
    for y in range(mat.shape[1]):       # 0输出矩阵的行数，1输出矩阵的列数
        p, x = np.histogram(mat[:, y], bins=n)  # np.histogram是一个生成直方图的运算 bins指定统计的区间个数；mat[]待统计数据的数组
        x = x[:-1] + (x[1] - x[0]) / 2          # x是什么？
        ax.plot(x, np.cumsum(p / mat.shape[0]))
    ax.set_xlabel('Accuracy', fontsize=10)
    ax.set_ylabel('CDF', fontsize=10)
    ax.grid()                           # grid用于设置绘图区网格线
    fig.tight_layout()                  # 自动调整子图参数，使之填充整个图像区域
    return fig


class MyPrettyPrinter(pprint.PrettyPrinter):  # 打印特定时刻的状态
    def format_namedtuple(self, object, stream, indent, allowance, context, level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write(object.__class__.__name__ + '(')
        object_dict = object._asdict()
        length = len(object_dict)
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            inline_stream = StringIO()
            self.format_namedtuple_items(object_dict.items(), inline_stream, indent, allowance + 1, context, level,
                                         inline=True)
            max_width = self._width - indent - allowance
            if len(inline_stream.getvalue()) > max_width:
                self.format_namedtuple_items(object_dict.items(), stream, indent, allowance + 1, context, level,
                                             inline=False)
            else:
                stream.write(inline_stream.getvalue())
        write(')')

    def format_namedtuple_items(self, items, stream, indent, allowance, context, level, inline=False):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write  # 在派生类中重写时，向当前流中写入字节序列
        last_index = len(items) - 1
        if inline:
            delimnl = ', '
        else:
            delimnl = ',\n' + ' ' * indent
            write('\n' + ' ' * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key + '=')
            self._format(ent, stream, indent + len(key) + 2,
                         allowance if last else 1,
                         context, level)
            if not last:
                write(delimnl)

    def _format(self, object, stream, indent, allowance, context, level):
        # We dynamically add the types of our namedtuple and namedtuple like
        # classes to the _dispatch object of pprint that maps classes to formatting methods
        # We use a simple criteria (_asdict method) that allows us to use the
        # same formatting on other classes but a more precise one is possible
        if hasattr(object, '_asdict') and type(object).__repr__ not in self._dispatch:
            self._dispatch[type(object).__repr__] = MyPrettyPrinter.format_namedtuple
        super(MyPrettyPrinter, self)._format(object, stream, indent, allowance, context, level)


NAMETUPLE_PRINTER = MyPrettyPrinter(indent=2, depth=10)


class MySummaryWriter(SummaryWriter):  # 继承summarywriter
    scalar_filter_list = None

    def add_text(self, tag, text_string, global_step=None, walltime=None):  # 将文本数据添加到summary
        text_string = string_to_tb_string(text_string)   # 将字符串转换成适合tensorboard 的格式，清晰易读
        SummaryWriter.add_text(self, tag, text_string, global_step=global_step, walltime=walltime)   # walltime 墙上时间

    def add_text_of_object(self, tag, object, global_step=None, walltime=None):  # 添加路径操作
        text_string = NAMETUPLE_PRINTER.pformat(object)  # pformat:返回列表或字符串当中的内容，保存为一个py文件，以便将来读取和使用，它并不会像pprint那样直接输出
        print(text_string)
        text_string = string_to_tb_string(text_string)
        SummaryWriter.add_text(self, tag, text_string, global_step=global_step, walltime=walltime)

    def set_scalar_filter(self, scalar_filter_list):     # 标量过滤器列表
        self.scalar_filter_list = scalar_filter_list

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):  # tag数据标识符；s_v保存的数值（纵坐标y);g_s时间步数（横坐标x）
        if self.scalar_filter_list is None:
            SummaryWriter.add_scalar(self, tag, scalar_value, global_step=global_step, walltime=walltime)
        elif any(s in tag for s in self.scalar_filter_list):
            SummaryWriter.add_scalar(self, tag, scalar_value, global_step=global_step, walltime=walltime)


class LearningLogger():   # 175
    def __init__(self):
        self.__tb_logger_is_set = False
        self.__tb_logger = None

        self.__path_folder = None
        self.__exp_name = None
        self.__lock = Lock()

        self.set_log_path(TMP_LOG_PATH, 'test', 'test')

    def set_log_path(self, path, log_folder_name, experiment_name):  # 路径初始化
        # assert os.path.exists(path)
        # assert os.access(os.path.dirname(path), os.W_OK)  # path用来检测是否有访问权限的路径；mode为F_OK，测试存在的路径；
    # F_测试path是否存在，R_测试是否可读，W_测试是否可写，X_测试是否可执行
    # os.path.dirname去掉文件名，返回目录
        assert isinstance(log_folder_name, str)  # 来判断一个对象是否是已知的类型（str）
        assert isinstance(experiment_name, str)

        self.__path_folder = os.path.join(path, log_folder_name, experiment_name + "-" + get_current_time_str())  # 连接
        self.__exp_name = experiment_name
        self.__tb_logger = MySummaryWriter(log_dir=self.__path_folder, filename_suffix="." + experiment_name)
        self.__tb_logger_is_set = True

    def get_tb_logger(self) -> MySummaryWriter:  # python函数定义的函数名后面，为函数添加元数据,描述函数的返回类型
        '''
        example
             GLOBAL_LOGGER.get_tb_logger().add_scalar('rlc_r'+str(self.id), self.n_step,self.n_step)
        :return: the global summary writer
        '''
        assert self.__tb_logger is None or self.__tb_logger_is_set is True, "Set log path first"
        return self.__tb_logger

    def close_logger(self):
        assert self.__tb_logger is None or self.__tb_logger_is_set is True, "Set log path first"
        self.__tb_logger.close()

    def get_log_path(self):
        return self.__path_folder + '/'

    def reset_event_file(self):
        sl = None
        if self.__tb_logger_is_set:
            sl = self.__tb_logger.scalar_filter_list
        self.__tb_logger = MySummaryWriter(log_dir=self.__path_folder, filename_suffix="." + self.__exp_name)
        self.__tb_logger.set_scalar_filter(sl)
        self.__tb_logger_is_set = True


GLOBAL_LOGGER = LearningLogger()  # 126行


class TBScalarToCSV():    # 转化成纯文本文件
    def __init__(self, events_out_path, csv_out_path, list_of_scalar_name: list):
        self.events_out_path = events_out_path
        self.csv_out_path = csv_out_path
        self.list_of_scalar_name = list_of_scalar_name
        assert isinstance(list_of_scalar_name, list), 'add a list of the scalar name'
        print(self.list_of_scalar_name)

        # for path, dirs, files in os.walk(self.events_out_path):
        # 	for filename in files:
        # 		if fnmatch.fnmatch(filename, 'events.out*'):
        # 			print(filename)
        # path_to_file = os.path.join(self.events_out_path,filename)
        SIZE_GUIDANCE = {
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,  # 训练中记录的图像
            event_accumulator.AUDIO: 4,   # 训练中记录的音频
            event_accumulator.SCALARS: 0, # 训练过程中的统计数据：最值、均值等
            event_accumulator.HISTOGRAMS: 1,  # 生成直方图
            event_accumulator.TENSORS: 10,
        }
        ea = event_accumulator.EventAccumulator(self.events_out_path, size_guidance=SIZE_GUIDANCE)
        # 初始化EventAccumulator对象
        path_to_dir = os.path.join(self.csv_out_path, 'csv')
        self._export_get_scalers(ea, path_to_dir)

    def _export_get_scalers(self, ea, to_dir):                   # 不是很懂
        ea.Reload()                             # 这一步是必需的，将事件的内容都导进去
        print('Reload', ea.Tags())              # tensorboard可以保存Image scalar等对象
        if not os.path.exists(to_dir):
            os.mkdir(to_dir)
        for ss in ea.Tags()['scalars']:         # 用户可以为每篇文章、图片、或信息添加一个或多个标签，从而根据标签进行分类
            # for ss in self.list_of_scalar_name:
            i = 0
            if any(s in ss for s in self.list_of_scalar_name):
                path_to_file = os.path.join(to_dir, ss + '.csv')
                with open(path_to_file, 'w') as file:
                    w = csv.writer(file, delimiter=',')
                    #w.writerows([(str(step.wall_time), str(step.step), str(step.value)) for step in ea.Scalars(ss)])
                    for step in ea.Scalars(ss):
                        i += 1
                        if i % 1 == 0:  #这里是步数
                            w.writerows([(str(step.wall_time), str(step.step), str(step.value))])



class TBTextToConfig():
    def __init__(self, events_out_path, config_out_path):
        self.events_out_path = events_out_path
        self.config_out_path = config_out_path

        for path, dirs, files in os.walk(self.events_out_path):  # os.walk用于通过在目录树中游走输出在目录中的文件名，向上或者向下
            for filename in files:
                if fnmatch.fnmatch(filename, 'events.out*'):  # fnmatch主要用于文件名称的匹配，其能力比简单的字符串匹配更强大
                    print(filename)                           # event代表某一个batch的日志记录
                    path_to_file = os.path.join(path, filename)
                    SIZE_GUIDANCE = {
                        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                        event_accumulator.IMAGES: 4,
                        event_accumulator.AUDIO: 4,
                        event_accumulator.SCALARS: 1,
                        event_accumulator.HISTOGRAMS: 1,
                        event_accumulator.TENSORS: 0,
                    }
                    ea = event_accumulator.EventAccumulator(path_to_file, size_guidance=SIZE_GUIDANCE)
                    path_to_dir = os.path.join(config_out_path, 'config')
                    self._export_text(ea, path_to_dir)

    def _export_text(self, ea, to_dir):
        ea.Reload()
        print('Reload', ea)
        if not os.path.exists(to_dir):
            os.mkdir(to_dir)

        for s in ea.Tags()['tensors']:
            if '/text_summary' in s:
                config_name = s.replace('/text_summary', '')
                path_to_file = os.path.join(to_dir, config_name + '.config')
                with open(path_to_file, 'w') as file:
                    for i in range(len(ea.Tensors(s))):
                        for ii in range(len(ea.Tensors(s)[i].tensor_proto.string_val)):
                            config_string = ea.Tensors(s)[i].tensor_proto.string_val[ii].decode('utf-8')
                            print(config_string)
                            file.write(config_string)


if __name__ == '__main__':

    for filename in os.listdir('../DDPG'):
        if fnmatch.fnmatch(filename, '*.py'):
            print(filename)
    # path = root.split(os.sep)
    # print((len(path) - 1) * '---', os.path.basename(root))
    # for file in files:
    # 	print(len(path) * '---', file)

    GLOBAL_LOGGER.set_log_path(TMP_LOG_PATH, "hello/", "test")
    # GLOBAL_LOGGER.get_tb_logger().set_scalar_filter(['hi'])

    GLOBAL_LOGGER.get_tb_logger().add_text("hi", "hi 5", 3)
    GLOBAL_LOGGER.get_tb_logger().add_scalar("hidfaas", 3)
    GLOBAL_LOGGER.get_tb_logger().add_scalar("aaa", 3)
    GLOBAL_LOGGER.close_logger()
