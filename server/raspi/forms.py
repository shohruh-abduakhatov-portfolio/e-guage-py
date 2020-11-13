from django import forms
import datetime

"""
{
    "img_0":{
        "id":"src/img/saved/20180918/img_20:09:26-1.jpg", "value":3.0
    }

}
"""


class GuageForm(forms.Form):
    id = forms.CharField(widget=forms.HiddenInput, required=False)
    value = forms.CharField(widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Value'}), label='Value', max_length=255, required=True)
    name = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Gauge Name'}), label='Uzb Name', max_length=255,required=True)
    nomenclature = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nomenclature'}), label='Nomenclature', max_length=255,required=True)
    date = forms.DateField(widget=forms.DateInput(attrs={'class': 'form-control has-feedback', 'id': 'single_cal3', 'onkeydown': 'return false'}, format='%m/%d/%Y'), label='Date', initial=datetime.date.today, required=True)
    time = forms.TimeField(widget=forms.TimeInput( attrs={'class': 'form-control has-feedback', 'id': 'single_cal3'}, format='%H:%M/%s'), label='Time', initial=datetime.datetime.now, required=True)
    gauge_type = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Gauge'}), label='Gauge', max_length=255,required=True)


    def as_dict(self, id=False):
        if id:
            return {
                'id': self['id'].value(),
                'value': self['value'].value(),
                'name': self['name'].value(),
                'nomenclature': self['nomenclature'].value(),
                'date': self['date'].value(),
                'time': self['time'].value(),
                'gauge_type': self['gauge_type'].value()
            }
        else:
            return {
                'value': self['value'].value(),
                'name': self['name'].value(),
                'nomenclature': self['nomenclature'].value(),
                'date': self['date'].value(),
                'time': self['time'].value(),
                'gauge_type': self['gauge_type'].value()
            }
