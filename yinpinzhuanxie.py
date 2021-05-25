import json

import requests
import time

APP_ID = '24098782' # 你申请的APP_ID
API_KEY = 'rv54ay89ZpYCSKSwTKBrHBZR' # 你申请的API_KEY
SECRET_KEY = 'p4UZIl0SKqvCure7Yb22341GNgwnbYEi' # # 你申请的SECRET_KEY

def get_access_token(cltid, srt_key): # 获取访问令牌
    # oauth_url = 'https://openapi.baidu.com/oauth/2.0/token'
    oauth_url = 'https://aip.baidubce.com/oauth/2.0/token'
    args_data = {'grant_type': 'client_credentials',
                 'client_id': cltid,
                 'client_secret': srt_key,
                 }
    cnt_type = {'Content-Type': 'application/json; charset=UTF-8'}
    resp = requests.post(oauth_url, data=args_data, headers=cnt_type)
    print("get baidu center info...")
    if resp.status_code != 200:
        print("have http error", resp.status_code)
        return None
    cnt = resp.json()  # 获取的内容变为字典
    cnt['expires_in'] += int(time.time())  # 将有效期时间记录
    with open('baidu.ck', 'w', encoding='utf-8') as fp:
        res = {'access_token': cnt['access_token'], 'expires_in': cnt['expires_in']}
        json.dump(res, fp)
    return cnt['access_token']

def upload_audio_file(access_token): # 上传你要识别的音频文件，得按照人家规定的参数和格式
    # speed_url = 'http://vop.baidu.com/server_api'
    speed_url = 'https://aip.baidubce.com/rpc/2.0/session/offline/upload/asr?access_token={%s}' % access_token
    # args_data = {'format': 'pcm',
    #              'rate': 8000,
    #              'channel': 1,
    #              'cuid': 'rocky_shop',  # 应用名称，可随意取名
    #              'token': atoken,
    #              }
    args_data = {
        "appId": APP_ID,
        "companyName": "百度", # 这个可以修改
        "callId": "20e59200-57da-423e-b613-6a8ce126d0a7", # 这个也可以改
        #　"agentFileUrl": "http://***", # 你语音文件的url，要公网可访问的地址，你写上之后可以自己先下载试一下
        "suffix": "wav",
    }
    header = {'Content-Type': 'application/json'}
    # resp = requests.post(speed_url, data=json.dumps(args_data), headers=header)
    with open(r'F:\QQ download\384759118\FileRecv\20210430_143052.m4a', 'rb') as filedata:
    # with open(r'C:\Users\38475\Documents\录音\录音 (3).m4a', 'rb') as filedata:
        resp = requests.post(speed_url, files={'file': filedata}, headers=header)
    info = resp.json()
    return info

def get_text_from_url(access_token): # 获取识别结果。免费用户是整点识别，上传之后得等待...
    datas = {
        "category": "OFFLINE_ASR_RESULT",
        "paras": {
            "appId": APP_ID,  # 百度云appId,必选
            "callId": "20e59200-57da-423e-b613-6a8ce126d0a7"  # 用户上传某一通的callId,必选
        }
    }
    data_url = 'https://aip.baidubce.com/rpc/2.0/search/info?access_token={%s}' % access_token
    header = {'Content-Type': 'application/json'}
    response = requests.post(data_url, data=json.dumps(datas), headers=header)
    information = response.json()
    return information

# 结果解析的东西自己随便写，如果提前不太清楚返回数据的格式，可以先整个都print出来，然后做相应的解析。至于识别结果的准确性，这个不好说...
access_token = get_access_token(API_KEY, SECRET_KEY)
print(access_token)
info = upload_audio_file(access_token)
print(info)
result = get_text_from_url(access_token)
print(result)
# tmp = result['data']['content']
# print(tmp)
# tmp_list = eval(tmp)
# for x in tmp_list:
#     print(x['sentence'])

