from django import forms
from .models import Student


class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ['name', 'reg_number', 'department']


class CaptureForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        label='Student Name',
        widget=forms.TextInput(attrs={'placeholder': 'Enter your name'})
    )
