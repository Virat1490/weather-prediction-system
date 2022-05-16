import urllib.request as urllib2
import json
import time

def get_value():
    TS = urllib2.urlopen("http://api.thingspeak.com/channels/%s/feeds/last.json?api_key=%s" % (1735963,'HH83CACNO1NSZ7YV'))
    response = TS.read()
    data=json.loads(response)
    H = data['field1']
    T = data['field2']
    A = data['field3']
    P = data['field4']
    I = data['field5']
    R = data['field6']
    print (H + "    " + T+ "    " + R + "    " + I + "    " + P + "    " +A)
    TS.close()
    return H,T,R,I,P,A
get_value()
