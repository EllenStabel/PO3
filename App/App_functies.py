from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button




class TitleScreen(Screen):
    pass


class MainScreen(Screen):
    pass


class ECGScreen(Screen):
    pass


class PPGScreen(Screen):
    pass


class EDAScreen(Screen):
    pass


class MainWidget(Widget):
    pass

class BoxLayoutExample(BoxLayout):
    pass


class FilterApp(App):

    def build(self):
        # Create the screen manager
        sm = ScreenManager()
        sm.add_widget(TitleScreen(name='title'))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ECGScreen(name='ECG'))
        sm.add_widget(PPGScreen(name='PPG'))
        sm.add_widget(EDAScreen(name='EDA'))
        return sm

    pass

FilterApp().run()
