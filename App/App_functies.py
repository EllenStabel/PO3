from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.core.window import Window
import numpy as np
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal, misc
import csv


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
# plt.plot(ecg)
# plt.show()

x_vals = []
y_vals = []

index = count()




'''
x = [1, 2, 3, 4, 5]
y = [5, 12, 6, 9, 15]

plt.plot(x, y)
plt.xlabel("x-as")
plt.ylabel("y-as")

'''


class TitleScreen(Screen):
    pass


class MainScreen(Screen):
    def initialise_ecg():
        return misc.electrocardiogram()

    ecg = initialise_ecg()
    ecg = ecg[0:10000]

    sos_baseline, sos_powerline, sos_lowpass = initialise_filters_ecg(360, 0.5, 49, 51, 100, 4)
    # plt.plot(ecg)
    # plt.show()

    x_vals = []
    y_vals = []

    # index = count()
    k = 0

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
        Clock.schedule_interval(self.update_ecg_grafiek,1/20)
        return self.fig1

    def update_ecg_grafiek(self, *args):

        self.manager.get_screen('ECG').ids.grafiekECG.clear_widgets()
        self.k += 1
        data_pre_filter = ecg[: 10 * self.k + 10]
        data_post_filter = ecg_filter(data_pre_filter, sos_baseline, sos_powerline, sos_lowpass)

        x_vals = [i / 360 for i in range(len(data_post_filter))]
        plt.cla()
        plt.plot(x_vals, signal.detrend(data_post_filter))
        plt.xlim(x_vals[-1] - 1, x_vals[-1])
        plt.ylim(-3, 3)
        self.fig1 = plt.gcf()
        self.manager.get_screen('ECG').ids.grafiekECG.add_widget(FigureCanvasKivyAgg(self.fig1))
'''
    def heartbeat(self):
        self.k = 0
        data_pre_filter = ecg[: 10 * self.k + 10]
        data_post_filter = ecg_filter(data_pre_filter, sos_baseline, sos_powerline, sos_lowpass)
        x_vals = [i / 360 for i in range(len(data_post_filter))]

        heartbeat, peak_index, peak_amplitude = calculate_heartbeat(data_post_filter, 1, 2.5, round(60 / 220 * 360),
                                                                    x_vals)
        Clock.schedule_interval(self.update_ecg_waarde, 1 / 20)
        self.hartslag = str(heartbeat)

    def update_ecg_waarde(self, *args):
        self.k += 1
        data_pre_filter = ecg[: 10 * self.k + 10]
        data_post_filter = ecg_filter(data_pre_filter, sos_baseline, sos_powerline, sos_lowpass)
        x_vals = [i / 360 for i in range(len(data_post_filter))]
        heartbeat, peak_index, peak_amplitude = calculate_heartbeat(data_post_filter, 1, 2.5, round(60 / 220 * 360),
                                                                    x_vals)
        self.hartslag = str(heartbeat)




    def plotPPG(self):

        self.manager.get_screen('PPG').ids.grafiekPPG.add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def plotEDA(self):

        self.manager.get_screen('EDA').ids.grafiekEDA.add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def waardeECG(self):
        pass

    def waardePPG(self):
        pass

'''


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
