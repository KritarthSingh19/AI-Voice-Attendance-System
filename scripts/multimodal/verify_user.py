from backend.services.attendance_service import mark_attendance

from scripts.face.face_verifier import verify_face
from scripts.models.voice_verifier import verify_voice


print("\nAI Multimodal Verification Started")

face_name = verify_face()

if face_name is None:

    print("\nFace verification failed")
    exit()

voice_name = verify_voice()

print(f"\nFace Identity : {face_name}")
print(f"Voice Identity: {voice_name}")

if face_name == voice_name:

    print("\nMATCH SUCCESSFUL")

    mark_attendance(
        name=face_name,
        confidence=99.0,
        method="Multimodal"
    )

else:

    print("\nIDENTITY MISMATCH")
    print("ACCESS DENIED")