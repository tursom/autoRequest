import html
import os

import requests
import bs4

for (url, name) in (("https://api.bilibili.com/x/article/list/web/articles?id=330029&jsonp=jsonp", "毛泽东选集 第一卷"),
                    ("https://api.bilibili.com/x/article/list/web/articles?id=331197&jsonp=jsonp", "毛泽东选集 第二卷"),
                    ("https://api.bilibili.com/x/article/list/web/articles?id=334156&jsonp=jsonp", "毛泽东选集 第三卷"),
                    ("https://api.bilibili.com/x/article/list/web/articles?id=336027&jsonp=jsonp", "毛泽东选集 第四卷"),):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass
    for content in requests.get(url).json()["data"]["articles"]:
        # print(content)
        c = requests.get(f"https://www.bilibili.com/read/cv{content['id']}/?from=readlist").content
        soup = bs4.BeautifulSoup(c, "html.parser", from_encoding="utf-8")
        article = soup.find("div", {"id": "read-article-holder"})
        # print(article)
        # break
        with open(f"{name}/{content['title']}.txt", "wb") as out:
            article = str(article).replace("<p>", "").replace("</p>", "\n")
            out.write(bs4.BeautifulSoup(article, "html.parser", from_encoding="utf-8").text.encode())
