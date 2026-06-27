from pathlib import Path
import csv
from datetime import datetime

ATTENDANCE_FILE = Path("backend/data/attendance.csv")


def initialize_attendance_file():

    if not ATTENDANCE_FILE.exists():

        ATTENDANCE_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            ATTENDANCE_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "name",
                "timestamp",
                "status",
                "confidence",
                "method"
            ])


def already_marked_today(name):

    initialize_attendance_file()

    today = datetime.now().strftime("%Y-%m-%d")

    with open(
        ATTENDANCE_FILE,
        "r"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row["name"] == name
                and row["timestamp"].startswith(today)
            ):
                return True

    return False


def mark_attendance(
    name,
    confidence,
    method
):

    initialize_attendance_file()

    if already_marked_today(name):

        print(
            f"{name} already marked today"
        )

        return False

    with open(
        ATTENDANCE_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            name,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Present",
            confidence,
            method
        ])

    print(
        f"Attendance marked for {name}"
    )

    return True