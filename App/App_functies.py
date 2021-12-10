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
import decryption
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
while True and len(arr)< 10001:
    new_values = loop()
    # print(new_values)
    arr += new_values


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

# def initialise_ecg():
    # return misc.electrocardiogram()

ecg = arr[0:10000]

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
        '''
        data_pre_filter = ecg[: 10 * self.k + 10]
        data_post_filter = ecg_filter(data_pre_filter, sos_baseline, sos_powerline, sos_lowpass)
        x_vals = [i / 360 for i in range(len(data_post_filter))]
        plt.cla()
        plt.plot(x_vals, signal.detrend(data_post_filter))
        plt.xlim(x_vals[-1] - 1, x_vals[-1])
        plt.ylim(-3, 3)
        self.fig1 = plt.gcf()
        '''

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
        self.hartslag =
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
