from django.db import models


class Student(models.Model):
    name = models.CharField(max_length=100)
    reg_number = models.CharField(max_length=50)
    department = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Attendance(models.Model):
    STATUS_CHOICES = [
        ('Present', 'Present'),
        ('Absent', 'Absent'),
    ]

    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    time = models.TimeField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Present')

    class Meta:
        unique_together = ('student', 'date')
        ordering = ['-date', '-time']

    def __str__(self):
        return f"{self.student.name} - {self.date} - {self.status}"
