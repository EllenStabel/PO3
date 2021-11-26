from kivy.app import App
from kivy.lang import Builder
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.core.window import Window
import matplotlib.pyplot as plt
import numpy as np
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

x = [1, 2, 3, 4, 5]
y = [5, 12, 6, 9, 15]

plt.plot(x, y)
plt.xlabel("x-as")
plt.ylabel("y-as")


class TitleScreen(Screen):
    pass


class MainScreen(Screen):
    def plotECG(self):
        self.manager.get_screen('ECG').ids.grafiekECG.add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def plotPPG(self):
        self.manager.get_screen('PPG').ids.grafiekPPG.add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def plotEDA(self):
        self.manager.get_screen('EDA').ids.grafiekEDA.add_widget(FigureCanvasKivyAgg(plt.gcf()))


class ECGScreen(Screen):
    pass


class PPGScreen(Screen):
    pass


class EDAScreen(Screen):
    pass


class ScreenManagement(ScreenManager):
    pass


class image(Image):
    pass


bestand = Builder.load_file("filter.kv")


class FilterApp(App):
    def build(self):
        return bestand


FilterApp().run()
