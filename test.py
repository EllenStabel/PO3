import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget


class BoxLayoutExample(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        b1 = Button()


class MainWidget(Widget):
    pass


class MonitorApp(App):
    pass
   #def build(self):
   #     return Label(text='Hello')


MonitorApp().run()