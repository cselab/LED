#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import time

# from datetime import datetime, timedelta

# def secondsToTimeStr(seconds):
#     sec = timedelta(seconds=int(seconds))
#     d = datetime(1,1,1) + sec
#     output_str = "%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second)
#     return output_str


def secondsToTimeStr(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        time_str = '%d[d]:%d[h]:%d[m]:%d[s]' % (days, hours, minutes, seconds)
    elif hours > 0:
        time_str = '%d[h]:%d[m]:%d[s]' % (hours, minutes, seconds)
    elif minutes > 0:
        time_str = '%d[m]:%d[s]' % (minutes, seconds)
    else:
        time_str = '%d[s]' % (seconds, )
    return time_str


# def secondsToTimeStr(seconds):
#     # Function to transform time in seconds to a str with hours, minutes and seconds.
#     print(time.gmtime(seconds))
#     return time.strftime('%d:%H:%M:%S', time.gmtime(seconds))
