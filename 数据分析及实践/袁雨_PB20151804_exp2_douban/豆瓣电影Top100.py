# -*- coding = utf-8 -*-
# @Time:2022/3/1712:06
# @Author:袁雨
# @File:豆瓣电影Top100.py
# @Software:PyCharm

import urllib.request, urllib.error     # 爬虫
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re   # 正则表达式，进行文字匹配
import json     # 存储
import time     # time.sleep()
import random   # 随机数
import os   # 文件操作


def main():
    baseurl = "https://movie.douban.com/top250?start="
    # 1.爬取网页
    datalist = getData(baseurl)
    # 2.逐一解析数据
    # 3.保存数据
    saveData(datalist)
    # 保存图片
    saveImg(datalist)


# 寻找规则
# 详情链接
findLink = re.compile(r'<a href="(.*?)">')     # 创建正则表达式对象，表示规则（字符串的模式）
# 片名
findTitle = re.compile(r'<span property="v:itemreviewed">(.*?)</span>')
# 导演
findDir = re.compile(r'<a href="/celebrity/(\d*)/" rel="v:directedBy">(.*?)</a>')
# 编剧
findWriter = re.compile(r'<a href="/celebrity/(\d*)/">(.*?)</a>')
# 主演
findAct = re.compile(r'<a href="/celebrity/(\d*)/" rel="v:starring">(.*?)</a>')
# 类型
findType = re.compile(r'<span property="v:genre">(.*?)</span>')
# 官方网站
findWeb = re.compile(r'<a href="(.*?)" rel="nofollow" target="_blank">(.*?)</a>')
# 制片国家/地区
findCountry = re.compile(r'<span class="pl">制片国家/地区:</span>(.*)<br/>')
# 语言
findLan = re.compile(r'<span class="pl">语言:</span>(.*)<br/>')
# 上映日期
findDate = re.compile(r'<span content="(.*?)" property="v:initialReleaseDate">(.*?)</span>')
# 片长
findRuntime = re.compile(r'<span class="pl">片长:</span> <span content="(.*?)" property="v:runtime">(.*?)</span>(.*)')
# 评分
findAver = re.compile(r'<strong class="ll rating_num" property="v:average">(.*)</strong>')
# 图片
findImg = re.compile(r'<img alt="(.*?)" rel="v:image" src="(.*?)" title="点击看更多海报"/>')


# 爬取网页
def getData(baseurl):
    datalink = []   # 保存详情链接
    for i in range(0, 4):  # 调用获取页面信息的函数 4次
        url = baseurl + str(i*25)
        html = askURL(url)  # 保存获取到的网页源码

        t = random.random()  # 随机大于0 且小于1 之间的小数
        time.sleep(t)

        # 2.逐一解析数据
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all('div', class_="item"):    # 查找符合要求的字符串，形成列表 类别 属性值 加下划线
            # print(item)   # 测试，查看电影item全部信息
            # 查找 添加
            # 链接
            item = str(item)
            link = re.findall(findLink, item)[0]     # re库用来通过正则表达式查找指定的字符串
            datalink.append(link)
    datalist = []   # 保存所有电影的信息
    for link in datalink:
        # print(link)
        html1 = askURL(link)    # 保存获取到的网页源码

        t = random.randint(0, 3)
        time.sleep(t)

        # 2.逐一解析数据
        soup = BeautifulSoup(html1, "html.parser")
        for content in soup.find_all(id='content'):  # 查找符合要求的字符串，形成列表 （类别 属性值 加下划线
            content = str(content)
            # print(content)  # 测试，查看电影content全部信息
            data = []  # 保存一部电影的所有信息
            # 查找 添加
            # 片名
            titles = re.findall(findTitle, content)[0]
            data.append(titles)
            # 导演
            director = re.findall(findDir, content)
            directors = []
            for i in range(len(director)):
                directors.append(director[i][1])
            data.append(directors)
            # 编剧
            writer = re.findall(findWriter, content)
            writers = []
            for i in range(len(writer)):
                writers.append(writer[i][1])
            data.append(writers)
            # 主演
            act = re.findall(findAct, content)
            acts = []
            for i in range(len(act)):
                acts.append(act[i][1])
            data.append(acts)
            # 类型
            typee = re.findall(findType, content)
            types = []
            for i in range(len(typee)):
                types.append(typee[i])
            data.append(types)
            # 官方网站
            web = re.findall(findWeb, content)
            if len(web) != 0:
                data.append(web[0][0])
            else:
                data.append(" ")
            # 制片国家/地区
            country = re.findall(findCountry, content)[0]
            country = re.sub('/', " ", country)  # 替换/
            data.append(country.strip())  # 去掉前后的空格
            # 语言
            lan = re.findall(findLan, content)[0]
            lan = re.sub('/', " ", lan)  # 替换/
            data.append(lan.strip())  # 去掉前后的空格
            # 上映日期
            date = re.findall(findDate, content)
            dates = []
            for i in range(len(date)):
                dates.append(date[i][1])
            data.append(dates)
            # 片长
            runtime = re.findall(findRuntime, content)
            runtimes = []
            runtimes.append(runtime[0][1])
            if runtime[0][2] != '<br/>':
                otherRuntime = runtime[0][2]
                otherRuntime = re.sub('<br/>', " ", otherRuntime)  # 去掉<br/>
                otherRuntime = re.sub('/', " ", otherRuntime)  # 替换/
                runtimes.append(otherRuntime.strip())
            data.append(runtimes)
            # 评分
            aver = re.findall(findAver, content)[0]
            data.append(aver)
            # 图片
            img = re.findall(findImg, content)[0][1]
            data.append(img)

            datalist.append(data)
    #     break
    # for da in datalist:   # 测试
    #      print(da)
    return datalist

# 得到指定一个URL的网页内容
def askURL(url):
    # 用户代理，表示告诉豆瓣服务器，我们是什么类型的机器、浏览器（本质上是告诉浏览器，我们可以接收什么水平的文件内容
    # 模拟浏览器头部信息，向豆瓣服务器发送消息
    # 头部信息 不多 用字典
    # head[""]

    #head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36"}
    head = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36'}
    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html =response.read().decode("utf-8")
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print('1')
            print(e.code)
        if hasattr(e, "reason"):
            print('2')
            print(e.reason)
    return html


#保存数据
def saveData(datalist):
    for i in range(len(datalist)):
        sample = {}
        sample["片名"] = datalist[i][0]
        sample["导演"] = datalist[i][1]
        sample["编剧"] = datalist[i][2]
        sample["主演"] = datalist[i][3]
        sample["类型"] = datalist[i][4]
        sample["官方网站"] = datalist[i][5]
        sample["制片国家/地区"] = datalist[i][6]
        sample["语言"] = datalist[i][7]
        sample["上映日期"] = datalist[i][8]
        sample["片长"] = datalist[i][9]
        sample["评分"] = datalist[i][10]
        # for key, value in sample.items():  # 测试
        #     print("%s:%s" % (key, value))
        file = open('result.json', 'a', encoding='utf8')
        file.write(json.dumps(sample, ensure_ascii=False))
        file.write('\n')
        file.close()


# 保存文件（图片、音视频等）
# 爬取并解析图片的路径，直接保存图片
def saveImg(datalist):
    saveDir = "./movie_poster/"  # 保存电影海报的路径，方便修改
    for i in range(len(datalist)):
        img_src = datalist[i][11]
        path_list = re.sub(r'[(:)(：)(·)]', "", datalist[i][0])      # 过滤特殊字符
        a = '_' + str(i+1)+'.jpg'
        # print(saveDir + path_list + a)    # 测试
        objPath = saveDir + path_list + a
        isExists = os.path.exists(saveDir)
        if not isExists:  # 如果不存在
            os.mkdir(saveDir)  # 创建文件夹
        img = urllib.request.urlopen(img_src)
        with open(objPath, "ab") as f:
            f.write(img.read())


if __name__ == "__main__":     # 如果执行的是主函数/主方法，  当程序执行时
    # 调用函数
    main()  # 作为整个程序的入口，从哪个函数开始执行
    print("爬取完毕！")


