from datetime import date, datetime
import csv
import io

from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import StudentForm, CaptureForm
from .models import Student, Attendance
from .face_recognition import run_recognition, capture_faces, train_recognizer, TRAINER_PATH
import os


def home(request):
    return render(request, "home.html")


def login(request):
    return render(request, "login.html")


def dashboard(request):
    now = datetime.now()
    total_students = Student.objects.count()
    today = date.today()
    now_time = now.strftime('%H:%M:%S')
    present_today = Attendance.objects.filter(date=today, status='Present').count()
    absent_today = total_students - present_today
    attendance_percentage = round((present_today / total_students) * 100, 2) if total_students else 0
    students = Student.objects.all().order_by('-created_at')
    recent_attendance = Attendance.objects.select_related('student').order_by('-date', '-time')[:20]
    last_attendance = recent_attendance.first()
    digital_features = [
        "Automatic face recognition attendance",
        "Real-time date and time logging",
        "Registered student details",
        "Attendance summary dashboard",
        "Future expansion: PDF/Excel export, SMS alerts, and analytics",
    ]

    model_trained = os.path.exists(TRAINER_PATH)
    context = {
        "student_count": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_percentage": attendance_percentage,
        "students": students,
        "recent_attendance": recent_attendance,
        "last_attendance": last_attendance,
        "digital_features": digital_features,
        "today": today,
        "now_time": now_time,
        "session_started_at": now.strftime('%Y-%m-%d %H:%M:%S'),
        "model_trained": model_trained,
    }
    return render(request, "dashboard.html", context)


def register_student(request):
    form = StudentForm(request.POST or None)

    if form.is_valid():
        form.save()
        return redirect('/dashboard/')
    return render(request, "register.html", {'form': form})


def reports(request):
    attendance_records = Attendance.objects.select_related('student').order_by('-date', '-time')
    total_records = attendance_records.count()
    return render(request, "reports.html", {
        'attendance_records': attendance_records,
        'total_records': total_records,
    })


def export_excel(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="attendance_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Name', 'Reg Number', 'Department', 'Date', 'Time', 'Status'])

    for record in Attendance.objects.select_related('student').order_by('-date', '-time'):
        writer.writerow([
            record.student.name,
            record.student.reg_number,
            record.student.department,
            record.date,
            record.time,
            record.status,
        ])
    return response


def export_pdf(request):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return HttpResponse('PDF export requires the reportlab package. Install it in your environment.', status=500)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 40
    y = height - 40

    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(x_margin, y, 'Attendance Report')
    y -= 30
    pdf.setFont('Helvetica', 10)
    pdf.drawString(x_margin, y, f'Total records: {Attendance.objects.count()}')
    y -= 24

    pdf.setFont('Helvetica-Bold', 10)
    pdf.drawString(x_margin, y, 'Name')
    pdf.drawString(x_margin + 140, y, 'Date')
    pdf.drawString(x_margin + 220, y, 'Time')
    pdf.drawString(x_margin + 290, y, 'Status')
    y -= 18
    pdf.setFont('Helvetica', 9)

    for record in Attendance.objects.select_related('student').order_by('-date', '-time'):
        if y < 70:
            pdf.showPage()
            y = height - 40
            pdf.setFont('Helvetica-Bold', 10)
            pdf.drawString(x_margin, y, 'Name')
            pdf.drawString(x_margin + 140, y, 'Date')
            pdf.drawString(x_margin + 220, y, 'Time')
            pdf.drawString(x_margin + 290, y, 'Status')
            y -= 18
            pdf.setFont('Helvetica', 9)

        pdf.drawString(x_margin, y, record.student.name)
        pdf.drawString(x_margin + 140, y, str(record.date))
        pdf.drawString(x_margin + 220, y, str(record.time))
        pdf.drawString(x_margin + 290, y, record.status)
        y -= 16

    pdf.save()
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="attendance_report.pdf"'
    return response


def train_model(request):
    message = None
    success = False
    trained = os.path.exists(TRAINER_PATH)

    if request.method == 'POST':
        result = train_recognizer()
        message = result.get('message')
        success = result.get('success', False)
        trained = success or trained

    return render(request, 'train.html', {
        'message': message,
        'success': success,
        'trained': trained,
    })


def capture(request):
    message = None
    message_type = 'info'
    count = None

    if request.method == 'POST':
        form = CaptureForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            result = capture_faces(name)
            message = result.get('message')
            message_type = 'success' if result.get('success') else 'error'
            count = result.get('count', 0)
    else:
        form = CaptureForm()

    return render(request, 'capture.html', {
        'form': form,
        'message': message,
        'message_type': message_type,
        'count': count,
    })


def recognize_face(request):
    started_at = request.GET.get('started_at') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result = run_recognition()
    if isinstance(result, dict):
        results = result.get('results', [])
        message = result.get('message')
    else:
        results = result
        message = None
    return render(request, 'recognize_result.html', {
        'results': results,
        'message': message,
        'session_started_at': started_at,
    })