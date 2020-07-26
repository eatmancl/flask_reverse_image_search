import datetime
import random


def create_uuid():
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");
    randomNum = random.randint(0, 100);
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum);
    uniqueNum = str(nowTime) + str(randomNum);
    return uniqueNum;
