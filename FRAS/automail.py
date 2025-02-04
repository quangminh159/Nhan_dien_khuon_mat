import yagmail
import os
import datetime
date = datetime.date.today().strftime("%B %d, %Y")
path = 'FRAS\Attendance'
os.chdir(path)
files = sorted(os.listdir(os.getcwd()), key=lambda x: os.path.getmtime(x))
if files:
    newest = files[-1]
    filename = newest
else:
    print("No files found in the 'Attendance' folder.")
    exit()
sub = "Attendance Report for " + str(date)
body = "Please find the attached attendance report for " + str(date)
receiver = "receiveremail@example.com"  
yag = yagmail.SMTP("youremail@email.com", "password")  
yag.send(
    to=receiver,
    subject=sub,  
    contents=body,  
    attachments=filename  
)

print("Email Sent!")
