from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import NumericProperty, ListProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
from scipy import signal, misc
import numpy as np
import requests
import decryption.py
import time
import random

api_url = "http://192.168.137.117:8080/api"

keyA_url = api_url + "/keyA/user/"
keyB_url = api_url + "/keyB/user/"

def limitTo64Bits(n):
    # Negeert overflow, zoals in cpp code
    c = 0xFFFFFFFFFFFFFFFF & n  # Gebruikt 64 bit
    return c

def sign_up(username, email, password):
    url = api_url + "/auth/signup"
    body = {"username": username, "email": email, "password": password}
    requests.post(url, data=body)


def get_token(username, password):
    # sign in
    url = api_url + "/auth/signin"
    body = {"username": username, "password": password}
    info = False

    while not info:  # Als info leeg is
        info = requests.post(url, data=body)
        print("Waiting for server response")
        time.sleep(5)

    info = info.json()

    user_id = info["id"]
    token = info["accessToken"]

    print("Authentication ok")

    return user_id, token


def postB_receiveA(B, user_id, access_token):
    A_url = keyA_url + str(user_id)
    B_url = keyB_url + str(user_id)

    obj = {'B': B}  # str(B)
    requests.post(B_url, data=obj, headers={"x-access-token": access_token})  # eerst posten

    response = requests.get(A_url, headers={"x-access-token": access_token})

    while response.status_code == 101:  # Table is nog leeg -> wachten tot values inkomen
        time.sleep(5)  # sec
        response = requests.get(A_url, headers={"x-access-token": access_token})

    A = response.json()["A"]  # is in json nog als string gestored
    A = int(A)

    return A


def askForKey(user_id, access_token):
    P = limitTo64Bits(2147483647)
    # P = 47251
    G = limitTo64Bits(2)
    #b = limitTo64Bits(random.getrandbits(64))  # random number genereren -> secret
    b = 3
    #B = limitTo64Bits(int(pow(G, b, P)))
    B = 8
    A = postB_receiveA(B, user_id, access_token)  # stuurt B door en ontvangt A
    s = limitTo64Bits(int(pow(A, b, P)))
    return s


def testSuite(user_id, access_token, soort):
    url = api_url + "/" + soort + "/user/" + str(user_id)
    response = requests.get(url, headers={"x-access-token": access_token})
    while response.status_code == 101:  # Table is nog leeg -> wachten tot values inkomen
        print("No new values")
        time.sleep(2)  # sec
        response = requests.get(url, headers={"x-access-token": access_token})

    print("Response received")
    response = response.json()

    plaintext = [response[i][soort] for i in range(len(response)) if response[i][soort] is not None]
    AANTAL = 32  # Aantal values per request

    se = []
    for j in range(0, len(plaintext)//AANTAL):
        res = []
        for i in range(j*AANTAL, (j+1)*AANTAL):
            res.append(plaintext[i])
        #print(len(res))
        se.append(res)
    # Zet plaintext die bij elkaar horen bij elkaar
    tags = [response[i]["tag"] for i in range(len(response)) if response[i]["tag"] is not None]
    tags_res = []
    for j in range(0, len(tags)//(2)):
        res = []
        for i in range(2):
            res.append(tags[i])
        tags_res.append(res)

    # plaintext = processData(response, soort)
    # return plaintext

    return se, tags_res


def processData(data, keyVal, tag):

    for i in range(len(data)):  # num is van de vorm "getal" zonder prefix 0x
        # data[i] = int(data[i], 16)  # Steekt in hexadecimale ull
        data[i] = int(data[i])

    tag = [int(x) for x in tag]  # decryptie algoritme verwacht het in int om te kunnen vergelijken
    decrypted_arr = decryption.decryption(data, keyVal, tag)  # data[i] is array met values die samen geencrypteerd zijn

    for i in range(len(decrypted_arr)):
        if decrypted_arr[i] is not None:
            decrypted_arr[i] = 5*decrypted_arr[i]/1024  # Naar volts
        else:
            decrypted_arr = []
    return decrypted_arr


k = True
l = 0
m = 0


def getKeyTest():
    global k, l, m
    k = True
    if k:
        l, m = get_token("test", "pw123")
    k = False
    return askForKey(l, m)


def loop():
    global k, l, m
    k = True
    if k:  # Eerste keer runnen, dan niet meer. Key generating moet ook nog hierin
        l, m = get_token("test", "pw123")
        # key = askForKey(l, m)
        key = 3000
    k = False
    data, tags = testSuite(l, m, "sat")

    result = []
    for i in range(len(tags)):
        arr = processData(data[i], key, tags[i])
        for j in range(len(arr)):
            result.append(arr[j])

    #result = processData(data, key, tags)

    return result



arr = []
while True:
    new_values = loop()
    print(new_values)
    # arr += new_values
constants = [0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f]

state = [0 for i in range(5)]
t = [0 for i in range(5)]


def limitTo64Bits(n):
    # Negeert overflow, zoals in cpp code
    c = 0xFFFFFFFFFFFFFFFF & n  # Gebruikt 64 bit
    return c


def rotate(x, l):
    temp = (x >> l) ^ (x << (64 - l))
    temp = limitTo64Bits(temp)
    return temp


def sbox(x):
    x[0] ^= x[4]
    x[0] = limitTo64Bits(x[0])
    x[4] ^= x[3]
    x[4] = limitTo64Bits(x[4])
    x[2] ^= x[1]
    x[2] = limitTo64Bits(x[2])
    t[0] = x[0]
    t[0] = limitTo64Bits(t[0])
    t[1] = x[1]
    t[1] = limitTo64Bits(t[1])
    t[2] = x[2]
    t[2] = limitTo64Bits(t[2])
    t[3] = x[3]
    t[3] = limitTo64Bits(t[3])
    t[4] = x[4]
    t[4] = limitTo64Bits(t[4])
    t[0] = ~t[0]
    t[0] = limitTo64Bits(t[0])
    t[1] = ~t[1]
    t[1] = limitTo64Bits(t[1])
    t[2] = ~t[2]
    t[2] = limitTo64Bits(t[2])
    t[3] = ~t[3]
    t[3] = limitTo64Bits(t[3])
    t[4] = ~t[4]
    t[4] = limitTo64Bits(t[4])
    t[0] &= x[1]
    t[0] = limitTo64Bits(t[0])
    t[1] &= x[2]
    t[1] = limitTo64Bits(t[1])
    t[2] &= x[3]
    t[2] = limitTo64Bits(t[2])
    t[3] &= x[4]
    t[3] = limitTo64Bits(t[3])
    t[4] &= x[0]
    t[4] = limitTo64Bits(t[4])
    x[0] ^= t[1]
    x[0] = limitTo64Bits(x[0])
    x[1] ^= t[2]
    x[1] = limitTo64Bits(x[1])
    x[2] ^= t[3]
    x[2] = limitTo64Bits(x[2])
    x[3] ^= t[4]
    x[3] = limitTo64Bits(x[3])
    x[4] ^= t[0]
    x[4] = limitTo64Bits(x[4])
    x[1] ^= x[0]
    x[1] = limitTo64Bits(x[1])
    x[0] ^= x[4]
    x[0] = limitTo64Bits(x[0])
    x[3] ^= x[2]
    x[3] = limitTo64Bits(x[3])
    x[2] = ~x[2]
    x[2] = limitTo64Bits(x[2])


def linear(st):
    temp0 = rotate(st[0], 19)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[0], 28)
    temp1 = limitTo64Bits(temp1)
    st[0] ^= temp0 ^ temp1
    st[0] = limitTo64Bits(st[0])
    temp0 = rotate(st[1], 61)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[1], 39)
    temp1 = limitTo64Bits(temp1)
    st[1] ^= temp0 ^ temp1
    st[1] = limitTo64Bits(st[1])
    temp0 = rotate(st[2], 1)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[2], 6)
    temp1 = limitTo64Bits(temp1)
    st[2] ^= temp0 ^ temp1
    st[2] = limitTo64Bits(st[2])
    temp0 = rotate(st[3], 10)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[3], 17)
    temp1 = limitTo64Bits(temp1)
    st[3] ^= temp0 ^ temp1
    st[3] = limitTo64Bits(st[3])
    temp0 = rotate(st[4], 7)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[4], 41)
    temp1 = limitTo64Bits(temp1)
    st[4] ^= temp0 ^ temp1
    st[4] = limitTo64Bits(st[4])


def add_constant(st, i, a):
    st[2] = st[2] ^ constants[12 - a + i]
    st[2] = limitTo64Bits(st[2])


def p(st, a):
    for i in range(0, a):
        add_constant(st, i, a)
        sbox(st)
        linear(st)


def initialization(st, key):
    p(st, 12)
    st[3] ^= key[0]
    st[3] = limitTo64Bits(st[3])
    st[4] ^= key[1]
    st[4] = limitTo64Bits(st[4])


def decrypt(st, length, plaintext, ciphertext):
    plaintext[0] = ciphertext[0] ^ st[0]
    plaintext[0] = limitTo64Bits(plaintext[0])
    for i in range(1, length):
        p(st, 6)
        plaintext[i] = ciphertext[i] ^ st[0]
        plaintext[i] = limitTo64Bits(plaintext[i])
        st[0] = ciphertext[i]
        st[0] = limitTo64Bits(st[0])


def finalization(st, key):
    st[0] ^= key[0]
    st[0] = limitTo64Bits(st[0])
    st[1] ^= key[1]
    st[1] = limitTo64Bits(st[1])
    p(st, 12)
    st[3] ^= key[0]
    st[0] = limitTo64Bits(st[0])
    st[4] ^= key[1]
    st[1] = limitTo64Bits(st[1])


def decryption(to_decrypt1, keyVal=3000, tag_given=[]): # to_decrypt1 ipv data_to_decrypt
    nonce = [limitTo64Bits(2000), limitTo64Bits(0)]  # 128 bits in totaal
    # nonce = [limitTo64Bits(2000) for i in range(len(ciphertext))]
    key = [limitTo64Bits(keyVal), limitTo64Bits(0)]  # 128 bits in totaal
    # nonce = [limitTo64Bits(2000) for i in range(len(ciphertext))]

    # plaintext max 2^64 blocks -> 2^67 bytes
    IV = 0x80400c0600000000

    # to_decrypt1 = 0x4a568ec0314375ac
    # to_decrypt2 = 0x2d11864b7ba223da

    ciphertext = to_decrypt1
    # ciphertext = data_to_decrypt Demodag maar 1 value die doorgestuurd wordt
    # plaintext = [0 for i in range(10)]
    plaintext = [0 for i in range(len(ciphertext))]

    state[0] = IV
    state[1] = key[0]
    state[2] = key[1]
    state[3] = nonce[0]
    state[4] = nonce[1]

    for i in range(len(state)):
        state[i] = limitTo64Bits(state[i])

    initialization(state, key)
    # decrypt(state, 2, plaintext, ciphertext)  # 2 is hoeveel values er tegelijk worden decrypt
    decrypt(state, len(ciphertext), plaintext, ciphertext)
    # print("Plaintext: " + hex(plaintext[0])+" "+hex(plaintext[1]))
    '''
    t = "Plaintext: "
    for i in range(len(plaintext)):
        t += hex(plaintext[i]) + " "
    print(t)
    '''
    finalization(state, key)
    '''
    #print("Tag: " + hex(state[3])+" "+hex(state[4]))
    r = "Tag: "
    for i in range(3, len(state)):
        r += hex(state[i]) + " "
    print(r)
    '''
    # Controle met Tag hier invoeren
    '''
    if state[3:] != tag_given:
        print("Calculated tag:", str(limitTo64Bits(state[3])), str(limitTo64Bits(state[4])))
        print("Tag given:", str(tag_given[0]), str(tag_given[1]))

    #return plaintext #, state[3], state[4]
    '''
    return plaintext

# ONS DEEL

def initialise_filters_ecg(sample_frequency, baseline_cutoff_frequency, powerline_cutoff_frequency_1,
                           powerline_cutoff_frequency_2, lowpass_cutoff_frequency, order):
    sos_baseline = signal.butter(order, baseline_cutoff_frequency, btype='high', output='sos', fs=sample_frequency)
    sos_powerline = signal.butter(order, [powerline_cutoff_frequency_1, powerline_cutoff_frequency_2], btype='bandstop',
                                  output='sos', fs=sample_frequency)
    sos_lowpass = signal.butter(order, lowpass_cutoff_frequency, btype='low', output='sos', fs=sample_frequency)
    return sos_baseline, sos_powerline, sos_lowpass


def initialise_filters_ppg(sample_frequency, ac_cutoff_frequency_1, ac_cutoff_frequency_2, dc_cutoff_frequency,
                           attenuation, order):
    sos_ac = signal.cheby2(order, attenuation, [ac_cutoff_frequency_1, ac_cutoff_frequency_2], btype='bandpass',
                           output='sos', fs=sample_frequency)
    sos_dc = signal.cheby2(order, attenuation, dc_cutoff_frequency, btype='low')
    return sos_ac, sos_dc


def initialise_filters_eda(sample_frequency, lowpass_cutoff_frequency, higpass_cutoff_frequency, order_lowpass, order_highpass):
    sos_lowpass = signal.butter(order_lowpass, lowpass_cutoff_frequency, btype='low', output='sos', fs=sample_frequency)
    sos_higpass = signal.butter(order_highpass, higpass_cutoff_frequency, btype='high', output='sos', fs=sample_frequency)
    return sos_lowpass, sos_higpass


def ecg_filter(unfilterd_signal, sos_baseline, sos_powerline, sos_lowpass):
    # assert len(unfilterd_signal) == length_signal
    unfilterd_signal = signal.sosfilt(sos_baseline, unfilterd_signal)
    unfilterd_signal = signal.sosfilt(sos_powerline, unfilterd_signal)
    filter_signal = signal.sosfilt(sos_lowpass, unfilterd_signal)
    return filter_signal


def ppg_filter(unfilterd_signal_red, unfilterd_signal_ir, sos_ac, sos_dc):
    # assert len(unfilterd_signal_red) == length_signal and len(unfilterd_signal_ir) == length_signal
    unfilterd_signal_red_ac = signal.sosfilt(sos_ac, unfilterd_signal_red)
    unfilterd_signal_ir_ac = signal.sosfilt(sos_ac, unfilterd_signal_ir)
    unfilterd_signal_red_dc = signal.sosfilt(sos_dc, unfilterd_signal_red)
    unfilterd_signal_ir_dc = signal.sosfilt(sos_dc, unfilterd_signal_ir)
    return unfilterd_signal_red_ac, unfilterd_signal_ir_ac, unfilterd_signal_red_dc, unfilterd_signal_ir_dc


def eda_filter(unfilterd_signal, sos_lowpass, sos_highpass):
    unfilterd_signal = signal.sosfilt(sos_lowpass, unfilterd_signal)
    filterd_signal = signal.sosfilt(sos_highpass, unfilterd_signal)
    return filterd_signal


def calculate_heartbeat(data_post_filter, min_peak, max_peak, sample_frequency, x_vals):
    peaks = signal.find_peaks(signal.detrend(data_post_filter), [min_peak, max_peak], distance=round(60 / 220 * sample_frequency))

    peak_index = peaks[0]
    peak_amplitude = peaks[1]['peak_heights']
    #heartbeat = 60 * len(peaks[0]) / x_vals[-1]
    if len(peak_index) >= 2:
        heartbeat = (peak_index[-1] - peak_index[-2]) / sample_frequency * 60
        return heartbeat, peak_index, peak_amplitude
    else:
        return 0, peak_index, peak_amplitude


def stress_detection_eda(data_post_filter, x_vals):
    data = data_post_filter[-1:-5]
    p = np.polyfit(x_vals, data, 1)
    y = np.polyval(p, x_vals)
    rico = p[0]
    if rico > 0:
        return 1
    else:
        return 0


# visualisatie

def initialise_ecg():
    return misc.electrocardiogram()

ecg = initialise_ecg()
ecg = ecg[0:10000]

sos_baseline, sos_powerline, sos_lowpass = initialise_filters_ecg(360, 0.5, 49, 51, 100, 4)

# EDA

eda = np.loadtxt(r'myData.txt')
sample_frequency = 100

sos_lowpass = signal.butter(6, 5, btype='low', output='sos', fs=sample_frequency)
sos_higpass = signal.butter(4, 0.05, btype='high', output='sos', fs=sample_frequency)

eda = signal.sosfilt(sos_lowpass, eda)
scr = signal.sosfilt(sos_higpass, eda)
time_scr = np.arange(0, len(scr) / sample_frequency, 1 / sample_frequency)
time_eda = np.arange(0, len(eda) / sample_frequency, 1 / sample_frequency)


class TitleScreen(Screen):
    def plot_ecg(self):
        self.k = 0

        data_pre_filter = ecg[: 10 * self.k + 10]
        data_post_filter = ecg_filter(data_pre_filter, sos_baseline, sos_powerline, sos_lowpass)
        x_vals = [i / 360 for i in range(len(data_post_filter))]
        plt.cla()
        plt.plot(x_vals, signal.detrend(data_post_filter))
        plt.xlim(x_vals[-1] - 1, x_vals[-1])
        plt.ylim(-3, 3)
        self.fig1 = plt.gcf()

        Clock.schedule_interval(self.update_ecg_grafiek, 1/20)
        # return self.fig1

    def update_ecg_grafiek(self, *args):
        self.manager.get_screen('ECG').ids.grafiekECG.clear_widgets()
        self.k += 1
        data_pre_filter = ecg[: 10 * self.k + 10]
        data_post_filter = ecg_filter(data_pre_filter, sos_baseline, sos_powerline, sos_lowpass)

        x_vals = [i / 360 for i in range(len(data_post_filter))]
        self.fig1 = plt.figure(1)
        plt.cla()
        plt.plot(x_vals, signal.detrend(data_post_filter))
        plt.xlim(x_vals[-1] - 1, x_vals[-1])
        plt.ylim(-3, 3)
        self.manager.get_screen('ECG').ids.grafiekECG.add_widget(FigureCanvasKivyAgg(self.fig1))

    hartslag = NumericProperty(0)

    def heartbeat(self):
        self.m = 0
        '''
        data_pre_filter = ecg[: 10 * self.m + 10]
        data_post_filter = ecg_filter(data_pre_filter, sos_baseline, sos_powerline, sos_lowpass)
        x_vals = [i / 360 for i in range(len(data_post_filter))]

        heartbeat, peak_index, peak_amplitude = calculate_heartbeat(data_post_filter, 1, 2.5, round(60 / 220 * 360),
                                                                    x_vals)
                                                                    '''
        Clock.schedule_interval(self.update_ecg_waarde, 1 / 20)
        '''
        self.hartslag = str(int(heartbeat))
        self.manager.get_screen('main').ids.waardeECG.text = str(self.hartslag)
        '''

    def update_ecg_waarde(self, *args):
        self.m += 1
        data_pre_filter = ecg[: 10 * self.m + 10]
        data_post_filter = ecg_filter(data_pre_filter, sos_baseline, sos_powerline, sos_lowpass)
        x_vals = [i / 360 for i in range(len(data_post_filter))]
        heartbeat, peak_index, peak_amplitude = calculate_heartbeat(data_post_filter, 1, 2.5, round(60 / 220 * 360),
                                                                    x_vals)

        self.hartslag = str(int(heartbeat))
        self.manager.get_screen('main').ids.waardeECG.text = str(self.hartslag)
        self.manager.get_screen('ECG').ids.waardeECG.text = str(self.hartslag)


    def plot_eda(self):
        self.h = 0
        '''
        data_post_filter = scr[: 100 * self.h + 100]
        x_vals = [i / 100 for i in range(len(data_post_filter))]
        plt.cla()
        plt.plot(x_vals, signal.detrend(data_post_filter))
        plt.xlim(x_vals[-1] - 1, x_vals[-1])
        plt.ylim(-3, 3)
        self.fig2 = plt.gcf()
        '''
        Clock.schedule_interval(self.update_eda_grafiek, 1 / 20)
        # return self.fig2

    def update_eda_grafiek(self,*args):
        self.manager.get_screen('EDA').ids.grafiekEDA.clear_widgets()
        self.h += 1
        data_post_filter = scr[: 100 * self.h + 100]
        x_vals = [i / 100 for i in range(len(data_post_filter))]
        self.fig2 = plt.figure(2)
        plt.cla()
        plt.plot(x_vals, signal.detrend(data_post_filter))
        plt.xlim(x_vals[-1] - 1, x_vals[-1])
        plt.ylim(-3, 3)
        self.manager.get_screen('EDA').ids.grafiekEDA.add_widget(FigureCanvasKivyAgg(self.fig2))

    def leeftijd_validatie(self):
        leeftijd = self.manager.get_screen('title').ids.input.text
        return leeftijd


class MainScreen(Screen):

    def plotPPG(self):

        self.manager.get_screen('PPG').ids.grafiekPPG.add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def plotEDA(self):
        pass

    def waardeECG(self):
        pass

    def waardePPG(self):
        pass
    color = ListProperty([1,1,1,1])

    def update_color(self):
        self.hartslag = calculate_heartbeat()[0]
        if int(self.hartslag) > int(130):
            self.color = [0,1,0,1]
        else:
            pass




class ECGScreen(Screen):
    def verwijderenECG(self):
        self.manager.get_screen('ECG').ids.grafiekECG.clear_widgets()


class PPGScreen(Screen):
    def verwijderenPPG(self):
        self.manager.get_screen('PPG').ids.grafiekPPG.clear_widgets()


class EDAScreen(Screen):
    def verwijderenEDA(self):
        self.manager.get_screen('EDA').ids.grafiekEDA.clear_widgets()


class ScreenManagement(ScreenManager):
    pass


class image(Image):
    pass


bestand = Builder.load_file("filter.kv")


class FilterApp(App):
    def build(self):
        return bestand


FilterApp().run()
