from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.core.window import Window

class TitleScreen(Screen):
    pass


class MainScreen(Screen):
    pass


class ECGScreen(Screen):
    print("hello")
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
