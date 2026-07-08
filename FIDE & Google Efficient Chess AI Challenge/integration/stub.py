import requests
from kaggle_environments import make



def network_check():
    # first let's make sure you have internet enabled
    print("인터넷 연결 확인 중")
    if requests.get('http://www.google.com',timeout=10).ok == True :
        return print("인터넷 연결 정상 확인")
    else : 
        return print("인터넷 연결을 확인해 주세요")

def set_up():
    # Now let's set up the chess environment!
    env = make("open_spiel_chess", debug=True)
    return env