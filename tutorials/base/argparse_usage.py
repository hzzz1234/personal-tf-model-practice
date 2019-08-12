# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="argparse usage")
    # parser.add_argument("echo")
    # args = parser.parse_args()
    # print(args.echo)
    # action="store_true" -v没有指定任何参数也可，其实存的是True和False，如果出现，则其值为True，否则为False

    # parser.add_argument("-v", "--verbosity", help="increase output verbosity",action="store_true")
    # args = parser.parse_args()
    # if args.verbosity:
    #     print("verbosity turned on")

    # parser = argparse.ArgumentParser()
    # # 指定类型转换
    # parser.add_argument('x', type=int,help="the base")
    # args = parser.parse_args()
    # answer = args.x ** 2
    # print(answer)


    # parser = argparse.ArgumentParser()
    # # help用来自定义帮助信息
    # parser.add_argument("square", type=int,
    #                     help="display a square of a given number")
    # # 指定取值范围
    # parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
    #                     help="increase output verbosity")
    # args = parser.parse_args()
    # answer = args.square ** 2
    # if args.verbosity == 2:
    #     print("the square of {} equals {}".format(args.square, answer))
    # elif args.verbosity == 1:
    #     print("{}^2 == {}".format(args.square, answer))
    # else:
    #     print(answer)

    parser = argparse.ArgumentParser(description="calculate X to the power of Y")
    # 定义互斥组,2个参数只能出现一个
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("x", type=int, help="the base")
    parser.add_argument("y", type=int, help="the exponent")
    # 解析出前面是已知的参数，后面是未知参数
    args, unparsed = parser.parse_known_args()
    answer = args.x ** args.y

    if args.quiet:
        print(answer)
    elif args.verbose:
        print("{} to the power {} equals {}".format(args.x, args.y, answer))
    else:
        print("{}^{} == {}".format(args.x, args.y, answer))


    print(args)
    print(unparsed)