#!/bin/bash
# when use gpushare and can't git/wget or time out
# reference :https://gpushare.com/docs/instance/network_turbo/

export https_proxy=http://turbo.gpushare.com:30000 http_proxy=http://turbo.gpushare.com:30000


# and when you finish git/wget please run the following command
#############################################
# unset http_proxy && unset https_proxy
#############################################