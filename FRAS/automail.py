import yagmail
import os
import datetime

# Get current date
date = datetime.date.today().strftime("%B %d, %Y")

# Define path to Attendance folder
path = 'FRAS\Attendance'
os.chdir(path)

# Get list of files sorted by modification time
files = sorted(os.listdir(os.getcwd()), key=lambda x: os.path.getmtime(x))

# Ensure there's at least one file in the directory
if files:
    newest = files[-1]
    filename = newest
else:
    print("No files found in the 'Attendance' folder.")
    exit()

# Email subject and body
sub = "Attendance Report for " + str(date)
body = "Please find the attached attendance report for " + str(date)

# Email receiver and credentials
receiver = "receiveremail@example.com"  # Replace with actual receiver email
yag = yagmail.SMTP("youremail@email.com", "password")  # Replace with your email and password

# Send the email with attachment
yag.send(
    to=receiver,
    subject=sub,  # Email subject
    contents=body,  # Email body
    attachments=filename  # File to attach
)

print("Email Sent!")
